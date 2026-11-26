# Prompting MiniMax H3 pi-H3 (2 or 4 steps, video with sound)

pi-H3 is MiniMax H3's FL2VA model distilled (Lakonik/pi-H3) to make the same kind of clip in **2 or 4 steps**. The
DWD cards (`minimax-h3-pi-2step`, `minimax-h3-pi-4step`) also skip H3's video decoder: the denoiser's last step makes
the frames. The prompt is H3's own - write it exactly as the MiniMax H3 guide says (the three fields, shots, dialogue
tags, the soundscape): the adapters change how many steps it takes, not what it understands.

```
integrated_multimodal_description: [Shot 1] <style>, <composition> ... <actions, camera, dialogue on a timeline>

overall_soundscape: <the sounds of the scene>

non_diegetic_music: <a score, or "None.">
```

## What it takes

| Input | pi-H3 | How |
|---|---|---|
| a prompt | yes | `prompt` |
| a first frame | yes | `first_frame` (a picture, or a clip: its last frame) |
| a last frame | yes | `last_frame` |
| reference pictures, clips, voices (`<Picture 1>`, `<Video 1>`, `<Audio 1>`: ref2va) | **no** - refused (the experimental `-ref2va` cards below take them) | the full `minimax-h3-ref2va` model |
| a motion guide, a carried sound, a soundtrack (`guide_clip`, `first_audio`, `soundtrack`) | runs (the same keyframe rows as a first frame), not checked against pi-H3's reference | |
| ControlNet, masked regeneration, MPEG-Diffusion, a refine pass | **no** - refused | the full models |
| extra LoRAs | run beside the adapter's | `loras` |

- **Steps are the adapter's**: 2 or 4 whatever the request asks; the sampler is LakonLab's Euler on its own schedule,
  guidance 1 (no negative prompt).
- **Length and size**: as H3 - frames 17k + 5 at 24 fps (up to ~15 s), sides multiples of 32, up to 1344x768.
- **Upscaling**: a DWD clip has no latents to enlarge before decoding - use the pixel upscaler (ESRGAN), the
  default; the latent upscaler is skipped.
- **Speed** (B70, from the prompt to the clip): 2 steps 832x480 x 5 s 32 s, x 10 s 43 s, 1344x768 x 5 s 63 s, x 10 s
  166 s; 4 steps about 1.3-1.8x that; H3 at 15 steps takes 3.6-6.6x the 2-step time (the B65 ~1.2-1.5x the B70).
- **Draft, then final**: a 2-step clip is a quick look; the 4-step one holds detail and motion better. For dialogue
  and close-ups the full `minimax-h3` (8-20 steps) is still the reference.

## Over Ref2VA (EXPERIMENTAL: `minimax-h3-pi-2step-ref2va`, `-4step-ref2va`)

**Fast drafts of reference shots; the subject holds, the camera, place pictures and carried guides drift.** The same
adapters loaded over H3's Ref2VA denoiser (they were trained on FL2VA's), so the reference inputs work: `<Picture i>`,
`<Video k>`, `<Audio j>` (`ref_images`, `ref_videos`, `ref_audios`) beside a first and a last frame and guides; ControlNet
and masked runs stay refused. Measured on production shots: the referenced subject (a ship, a singer) stays itself as
well as on Ref2VA; a locked camera pushes in, the details of a place picture go, a guide's carried frames are 2-6 dB
looser. 5.5x (2 steps) and 3.4x (4 steps) faster than Ref2VA at 15 steps. Use them to look at a shot fast; render the
film's clips with `minimax-h3-ref2va`.

Dialogue: lip sync holds - measured with SyncNet and Whisper on four talking shots (docs/ports/pi-h3.md "Lip sync"), every
pi card as in sync and as intelligible as its base at 15 steps; write the line in the prompt as for H3.

Everything in the MiniMax H3 guide about prompts (speech rate, sound, hands, negations, chaining clips) holds here.
