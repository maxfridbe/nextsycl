# Prompting MiniMax H3 (video with sound)

H3 makes 4-15 s of video **with its own sound** (speech, effects, music) from one prompt. It was trained on a
three-field structured prompt; free text works, but dialogue, timing and sound are far more reliable in this form.
The field names and their order are fixed:

```
integrated_multimodal_description: [Shot 1] <style>, <composition> ... <actions, camera, dialogue on a timeline>
[Shot 2] At 00:04.500, the camera cuts to ...

overall_soundscape: <ambient sound, the sounds of actions, non-verbal human sounds, for the whole clip>

non_diegetic_music: <a score the people in the scene cannot hear, or "None.">
```

## Rules
- **Shots**: `[Shot 1]` has no time. Every later shot starts `[Shot N] At MM:SS.mmm, the camera cuts to ...` with
  increasing times inside the clip. State the style (live-action / cinematic / animated), the composition, the
  setup, then the actions. At most ~3 shots in 10 s; one clear action per shot.
- **Dialogue**: give each speaker a stable id and the exact line in a tag, with the language:
  `the woman with a warm, low alto (S1) says: <d>[English] We leave at dawn.</d>`. Describe the voice in the
  speaker phrase (pitch, tone, emotion, accent). Languages: English, Chinese, Japanese, Korean, French, German,
  Spanish, Italian, Portuguese, Russian, Arabic.
- **Voiceover**: `... says in an off-screen voiceover ... while her lips remain completely closed.`
- **Camera**: name the move with its amplitude and speed - `The camera pushes in with small amplitude at slow
  speed.` Moves: push in, zoom out, pan, truck, tilt, pedestal, arc, tracking, static, handheld shake, POV, roll.
- **Sound vs music**: what is heard in the scene goes in `overall_soundscape` (and is rendered under the whole
  clip, so name only what should be heard); score goes in `non_diegetic_music` (`None.` for none).
- **Speech rate**: the model speaks for about 57% of any clip, and the word count - not the clip's length - sets
  how fast: 22 words is clear (3.3 words a second), 28 the most, 34 too fast. Size the clip to the line:
  words / 3.3 / 0.57 seconds (22 words ~ 12 s), 4-15 s. Leave a beat after a line so it is not cut off.
- **Sound**: `overall_soundscape` is rendered under the whole clip - weather, traffic or an electrical hum come
  back as hiss or music. Name only the sounds something in the shot makes.
- **Hold things in positive words**: blinking (say it), the shot size ("that exact shot size and camera position
  hold from the first frame to the last" - a "slight push in" compounds over chained clips), the light level and
  colour, an empty room (or extras re-roll).
- **No negations**: guidance is 1.0, so every word renders - "no cars" puts cars in. Say what is there.
- **Hands**: counts and plurals invite extras ("held in two hands ... the hands lift its lid" drew three), and an extreme
  close-up of hands distorts fingers. Keep hands to the wide and medium shots; for a macro insert show the object
  alone ("the lid swings open, the mechanism turns").
- **A reference picture's orientation sets the direction of travel**: a locomotive pictured from the front arrives however the
  prompt says "pulls away"; picture it from behind and say "pulls away from the camera, seen from behind, getting smaller".
- **A prop's picture fixes its kind, not its ornament**: counts of points, stripes or letters drift (five points became
  six to eight). Pick props with a bold simple shape or colour.
- **Do**: concrete visual and audio detail, physical descriptions of people (age, hair, clothes, posture).
  **Don't**: plot summaries, abstract adjectives, lines longer than the shot can hold.

## Keys that change what the prompt can say
- `first_frame` / `last_frame` (a picture): pinned at the first / last frame and shown to the text encoder as
  `<Picture 1>` / `<Picture 2>`: describe what happens from / to it.
- References (the Ref2VA model, `minimax-h3-ref2va`): `ref_images` (up to 9), `ref_videos` (3, their sound too),
  `ref_audios` (3), named in the prompt as `<Picture i>`, `<Video k>`, `<Audio j>` - pictures first, then clips (a
  clip's sound takes the next `<Audio j>` right before its `<Video k>`), then sounds:
  `The dancer from <Picture 1> moves as in <Video 1>, speaking in the voice of <Audio 2>.`
- Effects: `embedding:minimaxh3_bullet_time` (also art_is_explosion, blooming_flowers, dark_magic, fire_breath,
  four_seasons, kiss_camera, spiral_ascent, storm_magic, truman_show) where the effect should happen.
- `control_video` (canny / depth / HED / MLSD / pose frames) the clip follows; `control_mask` + `control_source`
  regenerate the white part of the mask: describe the whole result.
- Steps: 8 for a draft, 12-20 when dialogue or motion matter. Canvases: 768x576 is the house size; 4-15 s.

## Films: a recipe for each kind
A clip is 4-15 s; a film is several, each carrying on from the last, then joined (`nextsycl video join`, or your
own ffmpeg). What carries across a cut:
- `first_frame` - a picture (a PNG; a clip: its **last frame**, a still). The pose and the look, not the motion.
- `guide_clip: "<clip>.mp4:22:0"` - the clip's last 22 frames (5, 22, 39: 17k + 5) as one moving keyframe at
  frame 0: position *and* motion. Give no `first_frame` with it.
- `first_audio: "<clip>.mp4"`, `first_audio_s` - its sound tail (room tone, score, a voice mid-line) at frame 0.
- Whatever is carried, the next clip **begins with it again**: the join must overlap exactly that span (half
  trimmed each side, the sound cross-faded over all of it), or it plays twice. `nextsycl video join` reads it from
  the clips' records.

### A monologue, an interview, a dialogue (people talking to the camera)
The shot must hold still and the voice and face must stay the same over many clips.
- **Hub**: every clip opens *and* closes on clip 1's last frame (`first_frame: "first"`, `last_frame: "first"`):
  nothing compounds, any clip can be re-rendered. Each clip must end back in its opening pose.
- **No audio anchor**, a **2-frame** cross-fade at the joins (a 1 s anchor forces a 1 s overlap, which discards the
  pinned frames - heads jump). A steady voice: `ref_audios` (`<Audio 1>`) on every clip, and drop the timbre
  adjectives once the voice comes from it. Or a 22-frame guide of clip 1's end with its sound (0.9167 s).
- **Size each clip to its line**: 22 words is clear, 28 the most; clip = words / 3.3 / 0.57 s.
- **Lock everything in positive words**: "the camera sits in one fixed position on a tripod; that exact shot size
  holds from the first frame to the last", blinking, the eyeline (about three times), the light level, an empty room.
- `exposure_ref: "first"` (each pinned picture held to clip 1's luminance); one camera (two cut together jar).
- 768x576, 8 steps (12 when the lips matter), upscale 1.5; the realism LoRA at 0.5 for live-action people.
- Join: each clip levelled to -18 LUFS (one voice, one room: the bed must not swell at the cuts), fade out 1.5 s.
- `nextsycl video speech` does all of this from a text file.

### An action scene (one continuous shot: a creature rises, a building falls)
The motion must carry through the cuts, and the sound with it.
- **Chain** each clip from the last with the **motion guide and its sound**: `guide_clip: "<last>.mp4:22:0"`,
  `first_audio: "<last>.mp4"`, `first_audio_s: 0.9167`, no `first_frame`. A picture alone restarts the motion.
- **Plan in frames**: a clip is always 17k + 5 frames (10 s asks give 243), and each later clip replays the 22
  carried frames, so make it the 17k + 5 nearest its beat + 22 (260 frames = 10.833 s; ask exactly frames / 24)
  - 243, 260, 260 is a 29.96 s film. Join cutting the picture in the middle of the 22 frames.
- **Write each clip as a beat on its own timeline** (`At 00:03.000 ...`), opening "Continuing the shot, ..." and
  re-describing who is there (the text encoder sees only this clip's prompt). Keep the style sentence word for
  word in every clip; the first clip's can name the picture it starts from (`<Picture 1>`), the later ones
  cannot (a guide is not shown to the text encoder).
- One big action a clip, the camera static or one named move; name the impacts in `overall_soundscape`, the
  score's build in `non_diegetic_music`.
- 12 steps (motion), a square or wide canvas: 768x768 x 10 s fits INT8, a later clip's 10.8 s needs Q6_K
  (INT8's composition in 4 GiB less); 2x ESRGAN upscale.
- Join: level the **whole film** to -18 LUFS (levelling each clip would flatten a score that builds), fade out.
- `tests/api/kaiju_lighthouse.py` is this recipe end to end through the API.

### A montage, a music video (cuts between different shots)
- Clips stand alone (no guide, no audio anchor); a shared look comes from the same style sentence and, if
  wanted, the same `first_frame` picture or `ref_images` (`minimax-h3-ref2va`) for the subject.
- The music is the film's, not the clips': make each clip's `non_diegetic_music: None.` and lay a song (the
  audio kind, `minimax-music3`) under the joined cuts; a 2-frame cross-fade or plain cuts.

### A cut: what to carry, by kind (docs/video/scenes.md has the measurements)
| The cut | Use |
|---|---|
| the same shot goes on | `guide_clip` (22 frames) + `first_audio` |
| a new shot in the same scene, the sound goes on (music, room tone) | `first_audio` = the clip before, `first_audio_s 0.9167`, no `guide_clip`; on every clip after the first (without it the sound starts again from nothing) |
| the same person speaks again | `ref_images` + `ref_audios` + the picture as `first_frame` and `last_frame` - on **every** clip of the person, the first too (without it the model draws a voice, a man's for a woman) |
| the speaker changes | that person's `ref_images` + `ref_audios`; **one speaker a shot** (two in one shot put a voice on the wrong mouth) |
| the same person, a new place | `ref_images [person's character sheet, the place's picture]`: "the woman from the character sheet `<Picture 1>` stands here, on the quay of `<Picture 2>`" |
| a prop in the scene | `ref_images [person, ..., prop]`, "lifts the brass lantern `<Picture 3>`" |
| a new scene in the same look | `ref_images [a frame of the last scene]` "in the exact look and light of `<Picture 1>`", and the light in words |
| a match cut, a transformation | `first_frame` = the last frame + `last_frame` = the target |
| a seamless loop | `first_frame` = `last_frame` |
| another person does what a clip did | `ref_images [him]` + `ref_videos [clip]`, and the prompt names who is who: "Using the movement of the figure in `<Video 1>` only, the old man from `<Picture 1>` ..." |
| a voice over b-roll | `ref_audios [the voice]`, "in an off-screen voiceover ... while no one is on screen" |
| the same motion in another style | `control_video` (canny edges of the clip) - not with references |
| a fix | re-render the clip; `regen` only fills small gaps |

The voice reference is one steady 3 s+ recording per person (a speech model's built-in voice, or a designed one),
the same on every clip; a short reference - one clip's own line - did not hold a voice. Cuts between people are plain
cuts: nothing is carried, so nothing overlaps. `tests/api/harbour_dialogue.py` is this as a scene.

### Every kind
- **Draft small first** (384x384, 4 s, 8 steps: a whole film in minutes) to check the story, the prompts and the
  chain - not the picture: the noise is drawn over the latent's shape, so the final size is another composition.
- Never `cond_noise_aug` (it decodes as confetti) or `first_latent` to chain (it jumps in exposure).
- Memory grows with width x height x seconds: a clip that does not fit fails at the denoiser, and the studio
  then pauses its whole queue (`video.queue.resume`).
- docs/video/h3.md has why each of these is so.

## Example
```
integrated_multimodal_description: [Shot 1] Live-action, cinematic, a medium-wide shot frames a baker opening the
shutters of a small street bakery before sunrise. The camera pushes in with small amplitude at slow speed as the
middle-aged baker with a calm, slightly raspy voice (S1) places a fresh loaf on the counter and says:
<d>[English] First batch of the morning.</d> [Shot 2] At 00:05.000, the camera cuts to a close-up of steam rising
from the sliced bread.

overall_soundscape: Wooden shutters scrape open over a quiet street, trays clink inside, the crisp sound of bread
being sliced.

non_diegetic_music: A soft acoustic-guitar pattern at a moderate tempo with sparse upright-bass notes.
```
