# Scenes and the cuts between them

A film is scenes, and at each cut something must stay the same (the motion, a face, a voice, the look, the place) or
nothing must. Every key MiniMax H3 has for carrying something across is the right one for some kind of cut and the
wrong one for another. `tests/api/scene_boundaries.py` makes each kind of boundary every way the model offers
(25 clips, 384x384, 8 steps, ~25 min, through the API alone), with a contact sheet per clip and the numbers; this is
what it settled. The recipes are also in the guide the API serves (`video.describePromptGuideMD`). The general rules
of carrying (frame planning, joining, memory) are in [h3.md](h3.md).

Draft scale, one seed, one subject: read the table as "what worked, and what failed", not as measured margins.
Identity is judged from the sheets; the numbers are the seam (grey correlation of the carried frame, 1.0 = the same
picture) and the sound's mean spectral centroid (a crude timbre; the TTS voices were 3200 Hz and 2374 Hz).

| # | The cut | Carry | Keys (engine) | What it did |
|---|---|---|---|---|
| S1 | The same shot goes on (action, a move, a build) | motion + sound | `guide_clip "<A>:22:0"`, `first_audio`, `first_audio_s 0.9167`, no `first_frame` (INT8/Q6_K) | seam 1.0 at the carried frame; the run continues; the sound tail follows (30 s). A still (`first_frame` = A) also matched the picture but restarts the motion. `ref_videos [A]` + `ref_images [A's last frame]` (Ref2VA) works too, 2.7x the time |
| S2 | The same person speaks again | face, voice, camera | Ref2VA `ref_images [her picture]` + `ref_audios [her voice]` + `first_frame`/`last_frame` = the picture | face held in all five recipes; the voice only by the reference (centroid 2.7 kHz against the reference's 3.2; hub and guide 2.1-2.3 kHz). Pinning the picture as well keeps the shot size. A hub or a guide without a voice reference is fine for one scene, drifts over many |
| S3 | The speaker changes (reverse shot) | the other face and voice | Ref2VA `ref_images [his]` + `ref_audios [his]` | his face held (also from his picture as `first_frame`); his voice nearer the reference with the reference (2.5 against 2.1 kHz). Two people in one shot: `ref_images [a, b]` + `ref_audios [a, b]`, `<Picture 1>` / `<Picture 2>` named in the prompt - both faces held |
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

- **Pictures alone do not carry a voice, a clip alone does not carry a face.** A cut between people needs both
  `ref_images` and `ref_audios` named in the prompt (`<Picture i>`, `<Audio j>`, one for each person).
- **A reference clip pulls its own content in.** The picture reference is the weaker: say in words who is in the
  picture and that the clip supplies only the movement, or the clip's people appear.
- **References and first/last frames combine** (Ref2VA takes both); references and a control video do not.
- **One engine at a time**: the daemon swaps the denoiser (INT8/Q6_K to REF2VA) between jobs, so group a film's clips
  by engine when you can; the swap costs about a minute and the Ref2VA clip itself ~1.7x an INT8 one.
- **Seams are only exact where something is pinned**: a guide or a first frame gives 1.0; a reference alone
  (dialogue, a new place) is a hard cut - join it as one, with a 2-frame cross-fade and no overlap.
