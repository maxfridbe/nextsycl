# Scene tropes: scripts over 2-3 clips, judged across dimensions

[scenes.md](scenes.md) settled what each *kind of cut* needs (S1-S13). A film is not cuts but scenes people recognise:
the chase, the reveal, the shot / reverse-shot argument. Each trope below is a short script of 2 or 3 clips, written so
that what it needs to carry across the cuts is exactly what the model finds hard. Like `tests/api/harbour_dialogue.py`
(the shot / reverse-shot trope, T1), each would be one script that makes the whole scene through the API and scores it.

## The dimensions every trope is judged on

| | dimension | how it is measured |
|---|---|---|
| D1 | **Identity**: the same face, hair, build | contact sheets of each clip's frames, judged by eye (the seam number cannot tell people apart) |
| D2 | **Costume and props**: the coat, the lantern, the gun stays the same object | sheets; a prop that vanishes or changes is a fail |
| D3 | **Voice**: the same speaker sounds the same, two speakers differ | median pitch per clip (`harbour_dialogue.py`'s autocorrelation), against the reference |
| D4 | **Place**: the same room / street / quay | sheets; the establishing shot against the later ones |
| D5 | **Look and light**: time of day, colour, grade | mean luma and colour per clip against the first |
| D6 | **Motion continuity**: the move goes on across the cut, nothing restarts | the seam (grey correlation of the carried frame, 1.0 = the same picture) and the sheets |
| D7 | **Screen direction and eyeline** (the 180-degree rule): who is left, who looks where | sheets; flagged by hand |
| D8 | **Scale and framing**: the shot size changes as scripted (wide to close), the subject does not jump | sheets |
| D9 | **Sound bed**: the room tone / music / ambience carries or changes on purpose | the clips' sound; first_audio tail carried or not |
| D10 | **Performance**: the emotion goes the way the script says (calm to alarmed) | by eye; the weakest dimension to automate |

A trope passes when the dimensions it *needs* (the column below) hold, not all ten: a hard cut to another place is
meant to change D4.

## The list

Clips: how many the script takes. Needs: the dimensions that must hold. Recipe: the keys from scenes.md.

### Dialogue and people
| # | Trope | Script | Clips | Needs | Recipe | Risk |
|---|---|---|---|---|---|---|
| T1 | **Shot / reverse shot** (a conversation) | A speaks, B answers, A replies | 3 | D1 D3 D4 D7 | S2 + S3 (one speaker a shot, Ref2VA picture + voice each); `harbour_dialogue.py` | voices on the wrong mouths if two people share a shot |
| T2 | **Over-the-shoulder** | the same exchange, the listener's shoulder in the foreground | 2-3 | D1 D7 D8 | S3 with the place's picture | the foreground figure drifts into the speaker |
| T3 | **The two-shot to the close-up** | two people in a room; push in on one | 2 | D1 D2 D4 D8 | S1 (guide) then Ref2VA | the face changes as the framing tightens |
| T4 | **The monologue with cutaways** | she speaks; cut to what she describes; back | 3 | D1 D3 D4 | S2 voice carried through a b-roll cut (S10) | the voice drifts across the cutaway |
| T5 | **The phone call / split scene** | two people, two places, one conversation | 3 | D1 D3 D4 D5 (a different look each side) | S2 + S4 | each place must stay itself on every return |
| T6 | **The whisper / secret overheard** | someone listens at a door; the voices come through it | 2 | D1 D3 D9 | S3 + the sound bed | the muffled voice is the wrong voice |

### Action and movement
| # | Trope | Script | Clips | Needs | Recipe | Risk |
|---|---|---|---|---|---|---|
| T7 | **The chase** | a runner, the pursuer behind, a turn into an alley | 3 | D1 D2 D6 D7 | S1 (guide_clip + first_audio) each cut | the runner's costume changes; the pursuer swaps sides |
| T8 | **The fight** | a blow, the reaction, the counter | 3 | D1 D6 D7 D10 | S1, an effect beat (S7) on the hit | limbs and faces melt at speed |
| T9 | **The car pursuit** | the car out of the garage, onto the road, around a corner | 3 | D2 D4 D6 D7 | S1 | the car's colour and shape |
| T10 | **The leap / the fall** | a jump across a gap, the landing | 2 | D1 D6 D8 | S1 or match cut (S6) | the figure's scale changes mid-jump |
| T11 | **Running toward the camera, then past it** | a sprint that crosses the lens | 2 | D1 D6 D8 | S1 | the figure's size is not continuous |

### Discovery and suspense
| # | Trope | Script | Clips | Needs | Recipe | Risk |
|---|---|---|---|---|---|---|
| T12 | **The reveal** | she opens a door; we see what she sees | 2-3 | D1 D4 D7 D10 | S1 then a point-of-view cut with a Ref2VA picture of the thing | the thing is not what the script says |
| T13 | **The slow dolly-zoom (vertigo)** | the background stretches while the subject holds | 2 | D1 D8 | S1 | the effect reads as a plain zoom |
| T14 | **The mystery object** | a hand lifts a lantern; close on the lantern; she looks at it | 3 | D1 D2 D8 | S13 (a prop) with S1 | the lantern changes between shots |
| T15 | **The look off-screen** | she stares past the camera; cut to what she sees | 2 | D1 D7 D10 | S3-style with the other picture | she looks at the wrong side |
| T16 | **The ticking clock** | a face; the clock; the face again, tenser | 3 | D1 D2 D10 | S6 (match cut) | the clock's hands do not move forward |
| T17 | **The stalker's view** | a long lens on a figure walking away | 2 | D1 D6 D8 | S1 | the figure changes as it recedes |

### Place and time
| # | Trope | Script | Clips | Needs | Recipe | Risk |
|---|---|---|---|---|---|---|
| T18 | **The establishing shot, then the interior** | a harbour at dawn; the room above it | 2 | D4 D5 | S5 (the look) | the room does not belong to the harbour |
| T19 | **The time-lapse / passage of time** | the same room at dawn, noon, dusk | 3 | D4 D5 (changing on purpose) D2 | first_frame/last_frame (S6) | the room rearranges itself |
| T20 | **The flashback** | a face; dissolve to the younger face in another time | 2-3 | D1 (older then younger) D5 | S4 with the look changed in words | the younger face is a different person |
| T21 | **The montage** | several short beats, one idea, one grade | 3 | D5 D9 | S5 with shared words | the grade drifts |
| T22 | **The travel / arrival** | the train leaves; the same train, a new city | 2-3 | D2 D4 D6 | S1 then S4 | the train changes |
| T23 | **Match cut** | the bone to the spacecraft | 2 | D6 D8 | S6 | the shapes do not line up |

### Transformation and effect
| # | Trope | Script | Clips | Needs | Recipe | Risk |
|---|---|---|---|---|---|---|
| T24 | **The transformation** | a person, a flash, the same person transformed | 2 | D1 (the new form) D6 | S6, S7 | identity is lost not changed |
| T25 | **The slow-motion hero moment** | the leap, bullet time, the landing | 2-3 | D1 D6 | S7 (`embedding:minimaxh3_bullet_time`) | reads as a spin, not a freeze (scenes.md) |
| T26 | **The loop** | the same shot ends where it begins | 2 | D6 | S8 | near-static |
| T27 | **The double / the mirror** | she meets her reflection | 2 | D1 (two of her) D7 | Ref2VA with the same picture twice | the two are not both her |

### Genre staples
| # | Trope | Script | Clips | Needs | Recipe | Risk |
|---|---|---|---|---|---|---|
| T28 | **The heist briefing** | the team round a table, each speaks, a plan on the wall | 3 | D1 (several) D3 D4 | S3 per speaker, one speaker a shot | too many faces to hold |
| T29 | **The final stand** | the hero against a crowd, the camera circling | 3 | D1 D6 D7 | S1, S9 (motion transfer) | the crowd swallows the hero |
| T30 | **The farewell at the station** | two figures, the goodbye, the train away | 3 | D1 D3 D4 D10 | S2/S3, S1 | the emotion arc is not read |
| T31 | **The cold open** | a scene, then the title over a hard cut to another | 2 | D4 D5 (changes) D9 | plain cut (S4) | the join clicks (cut the sound to the picture) |
| T32 | **Parallel action** | two scenes intercut: the bomb, the hero's run | 3 | D1 D2 D4 D5 | S4 each time | each strand must stay itself on every return |

## What to pick

The scripts are cheap on paper and costly to render (a draft clip is ~1-2 minutes, Ref2VA ~1.7x; a 3-clip trope ~15 min).
Two that together span the dimensions, each with its own hard part:

- A **dialogue** trope (T1 is done: `harbour_dialogue.py`; T5, T28 and T30 are its relatives) stresses D1 D3 D7.
- An **action** trope (T7, T8, T9) stresses D1 D2 D6, the motion carried by `guide_clip`.
- A **suspense** trope (T12, T14, T16) stresses D2 D8 D10, the prop and the reveal.
- A **place and time** trope (T19, T20) stresses D4 D5.

## Chosen: T14 and T30

`tests/api/trope_scenes.py` makes and scores two of them, built like `harbour_dialogue.py` (pictures and voices as Ref2VA
references, one speaker a shot, each clip sized to its line):

- **T14, the mystery object** (D1 D2 D4 D5 D8 D10): a medium shot of the woman lifting a walnut music box out of a trunk in
  an attic, an extreme close-up of the lid opening on its mother-of-pearl star, a medium close-up of her listening. The
  box is one reference picture on all three clips; the hard part is the box staying the same object while the shot size
  changes twice. (The lantern in the table above became a music box, because it has a face to recognise and a sound.)
- **T30, the farewell at the station** (D1 D3 D4 D7 D9 D10): she speaks (facing right), he answers from the carriage door
  (facing left), the train pulls away with her alone on the platform waving. Two faces, two voices, one station, and the
  eyeline across the cut.

Scoring: the measured checks are the clips' size, length and sound, each speaker's pitch against the voice sample, and the
film being the clips cut together. The rest (identity, the prop, the place, the light, the shot size, screen direction,
the performance) are asked of a vision model, one question per dimension, 1-5, 4 passes, shown frames of the clips and
the reference pictures at temperature 0. The judge is a model and can be wrong: the frames it saw are written beside the
report (`frame_*.jpg`) with each clip's contact sheet, so a score can be argued with, and the by-eye review these
dimensions started with is still the final word.

### Results

**T14, the mystery object** (hardac1, `--quick`: 384x288, 8 steps, no upscale, INT8 + Ref2VA, seeds 1955-1957; the three clips took 61, 31 and 31 s, the pictures and the judge's model load the rest of ~25 min). All measured checks and all six judged
dimensions passed: identity 5, prop 5, place 5, light 5, scale 5, performance 4 (of 5). Reviewed by eye against the contact
sheets, the judge was a little generous on the prop: the reference box has a five-pointed star on its lid, the clips show a
six- or eight-pointed one, and the latch the picture shows reads as a hinge. It is the same kind of box, in the same wood and
light, but not the same inlay; a stricter question (count the star's points) would score it lower. The woman, the attic and the
grade held across all three shots, and the shot size went medium, macro, close as scripted. T30 is not run yet.

Two things the first run missed, found by looking and listening rather than by the judge:

- **Sound.** Each clip's sound was generated alone, so the melody did not go on across the cut and take 3 began near silent
  before a burst (-46 dB, then -17 dB). Every clip after the first now carries the sound of the one before it
  (`first_audio`, the last 22 frames' worth, no picture carried so a plain cut plays nothing twice): take 3 then holds -24 to
  -26 dB throughout. Whether it is the same melody is judged by ear.
- **Three hands.** The macro prompt said the box was "held in two hands" and then that "the hands lift its lid"; the model drew a
  third. The prompt now says the box rests in one open palm and the thumb of her other hand lifts the lid: two hands in
  every frame. The judge had passed the three-hands take on its own prompts (nothing asked about hands); anatomy
  questions (hands in each clip; one body each in T30) are now judged too, and they did flag the three-hands take 1/5, though
  they passed the first run's take 2, which on a second look also had an extra hand. A model judge misses things; look at
  the sheets.

The lid's star is still not the reference's: five points in the picture, six to eight in the clips.

**T30, the farewell at the station** (hardac1, `--quick`, sound carried across clips; her clip 60 s, his 40 s, the wide 30 s).
Measured: every check passed; her voice 216 Hz against the sample's 242, his 143 against 147 (her sound tail was carried into
his clip and no voice leaked), hers higher than his. Judged: her identity 5, his 5, the place 5, the light 5, the eyelines
across the cut 5, anatomy 5, the goodbye reads 5; **the train 2**. Looking at the sheets agrees on the people and
disagrees on the train: it is a green engine with red wheels but smaller and a different shape from the reference, and in
the wide it appears to **arrive** (small, then large) where the script says it pulls away, a thing the judge was not asked
(a question on the direction of travel is now). One flaw the judge did not see: a red blotch on his coat in the second clip.
The train is the hard part of this trope, as the reference picture holds the kind of engine, not its build or the
direction it goes.
