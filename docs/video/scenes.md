# Scenes and the cuts between them

A film is scenes, and at each cut something must stay the same (the motion, a face, a voice, the look, the place) or
nothing must. Every key MiniMax H3 has for carrying something across is the right one for some kind of cut and the
wrong one for another. `tests/api/scene_boundaries.py` makes each kind of boundary every way the model offers
(25 clips, 384x384, 8 steps, ~25 min, through the API alone), with a contact sheet per clip and the numbers; this is
what it settled. The recipes are also in the guide the API serves (`video.describePromptGuideMD`). The general rules
of carrying (frame planning, joining, memory) are in [h3.md](h3.md).

Draft scale, one seed, one subject: read the table as "what worked, and what failed", not as measured margins.
Identity is judged from the sheets; the numbers are the seam (grey correlation of the carried frame, 1.0 = the same
picture) and, for voices, the **median pitch** of the clip (`harbour_dialogue.py`'s autocorrelation check: enough to
tell a woman's voice from a man's, not a speaker from a speaker). The first version of this page judged voices by
the mean spectral centroid, which cannot tell them apart, and called a two-person shot good that put a voice on the
wrong mouth; both are corrected below.

| # | The cut | Carry | Keys (engine) | What it did |
|---|---|---|---|---|
| S1 | The same shot goes on (action, a move, a build) | motion + sound | `guide_clip "<A>:22:0"`, `first_audio`, `first_audio_s 0.9167`, no `first_frame` (INT8/Q6_K) | seam 1.0 at the carried frame; the run continues; the sound tail follows (30 s). A still (`first_frame` = A) also matched the picture but restarts the motion. `ref_videos [A]` + `ref_images [A's last frame]` (Ref2VA) works too, 2.7x the time |
| S2 | The same person speaks again | face, voice, camera | Ref2VA `ref_images [her picture]` + `ref_audios [her voice]` + `first_frame`/`last_frame` = the picture | face held in all five recipes; **the voice only by the reference**: with it her clips were 267-276 Hz (the reference 296 / 276); without it (hub, guide, a plain picture) the model drew a man's voice for her, 163-170 Hz, and a cut to a clip with the reference changed her voice. Give every clip of a person the reference, the first too |
| S3 | The speaker changes (reverse shot) | the other face and voice | Ref2VA `ref_images [his]` + `ref_audios [his]`, one speaker in the shot | his face held; his voice at 99-105 Hz against the reference's 118 (from his picture alone: 125, and not stable). **Two people in one shot** (`ref_images [a, b]` + `ref_audios [a, b]`): both faces held, but the voices land on the wrong mouths - one speaker a shot and cut |
| S4 | A hard cut to the same person elsewhere | identity (not the place) | Ref2VA `ref_images [her]`, the place in words; or `[her, the place]` | the outfit and look held; with only the place's picture the model invents a person. With both pictures the subject is small in a wide shot |
| S5 | A new scene in the same look (establishing shot) | the look | Ref2VA `ref_images [A's last frame]`, "in the exact look and light of `<Picture 1>`" | slightly nearer A's light (mean luma 90 against 83 for words; A 98) - weak; say the light and colour in words as well |
| S6 | A match cut, a transformation | a picture at each end | `first_frame` = A's last frame + `last_frame` = the target (INT8); or the guide + `last_frame` | arrives exactly on the target (a clock face) by the clip's last frame; the guide version keeps A's motion for its first second then dissolves |
| S7 | An effect beat in a scene | the scene + an effect | the guide as S1, `embedding:minimaxh3_bullet_time` where it happens | the leap and the turn happen; at 384 px the effect reads as a spin, not a freeze |
| S8 | A seamless loop | one picture at both ends | `first_frame` = `last_frame` = the picture | near-static, the loop closes; wanted motion needs the prompt to ask for it |
| S9 | Another person does what A did (motion transfer) | identity from a picture, motion from a clip | Ref2VA `ref_images [him]` + `ref_videos [A]` (`ref_video_sound false`) **and a prompt that names who is who**: "Using the movement of the figure in `<Video 1>` only, the white-bearded old man ... from `<Picture 1>` ... Every person on screen is that old man" | with the bare "moves as in `<Video 1>`" the clip's own woman came out, not him (the video dominates the picture). Described, he runs with A's composition and gait. The picture alone gives his identity and not the composition. A control video (canny of A) with his picture as `first_frame` also works but jumps from the picture's room |
| S10 | A voice over other pictures (b-roll) | a voice | Ref2VA `ref_audios [her voice]` + the place's picture, "in an off-screen voiceover ... while no one is on screen" | the place held and no one invented; whether the voice-over was spoken was not measured |
| S11 | The same motion in another style | composition + motion | `control_video` (canny edges of A: `ffmpeg -vf edgedetect`), the new style in the prompt (INT8; not with Ref2VA) | the watercolour follows A's layout and movement exactly |
| S12 | A fix inside a scene | the rest of the clip | `source`, `regen "FROM-TO"`, `regen_box "x0,y0,x1,y1"` | weak: a box asked for a red coat put a small red figure in the distance, and a regenerated tail came back as the original. Re-render the clip (a hub clip is independent) instead |

## What the cuts taught

- **Where the voice comes from** (`harbour_dialogue.py --voices`, quick drafts of the same scene, pitch of her/his
  clips against the reference): *sample* - one recording of a stock voice per person (3 s+): 258-276 Hz / 99-105 Hz
  against 276 / 118, held; *design* - one recording in a voice the design model made up from a description: 167 /
  107-116 against 168 / 128, held (the voices only 50 Hz apart: describe them far apart); *lines* - each clip's own
  line spoken by the speech model as its reference: **his 4-word line came out at 213 Hz**, a woman's. A short
  reference (a line of a second and a half) is not enough to hold a voice; send one steady sample per person, the
  same on every clip.
- **Size the clip to the line**: 11 words got 158 frames (words / 3.3 / 0.57 s); the first reel packed 16 words into
  4.5 s, the "too fast" rate the guide warns of.
- **A hard cut is a plain cut**: nothing is carried, so nothing is trimmed or cross-faded; cut the sound to the
  picture's exact length first (a clip's sound runs a fraction of a frame long and a concat holds a frame for it),
  6 ms in and out so it does not click, and level each clip alone.
- **Design the scene, do not collect the clips**: the first "reel" joined the best clip of each experiment, which
  were not one scene - different voices, a clip made without the reference beside one with it - and it was unwatchable.
  `harbour_dialogue.py` is the same recipes as a scene.

- **Pictures alone do not carry a voice, a clip alone does not carry a face.** A cut between people needs both
  `ref_images` and `ref_audios` named in the prompt (`<Picture i>`, `<Audio j>`, one for each person).
- **A reference clip pulls its own content in.** The picture reference is the weaker: say in words who is in the
  picture and that the clip supplies only the movement, or the clip's people appear.
- **References and first/last frames combine** (Ref2VA takes both); references and a control video do not.
- **One engine at a time**: the daemon swaps the denoiser (INT8/Q6_K to REF2VA) between jobs, so group a film's clips
  by engine when you can; the swap costs about a minute and the Ref2VA clip itself ~1.7x an INT8 one.
- **Seams are only exact where something is pinned**: a guide or a first frame gives 1.0; a reference alone
  (dialogue, a new place) is a hard cut - join it as one, with a 2-frame cross-fade and no overlap.
