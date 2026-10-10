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
- **Speech rate**: about 3.8 words a second (3.3 at the slowest). A clip holds roughly (seconds - 2) x 3.3 words:
  12 s ~ 34 words. Leave a beat after a line so it is not cut off.
- **No negations**: guidance is 1.0, so every word renders - "no cars" puts cars in. Say what is there.
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
