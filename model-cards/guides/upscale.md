# Upscaling: Real-ESRGAN, waifu2x, SeedVR2 and FlashVSR

`image.upscale` makes a picture bigger; `video.upscale` does it to every frame of a clip (the sound copied as it was)
and answers an MP4. Both are jobs (wait, reentrant_id, `image.status`, `image.cancel`).

| model | variant | scales | for |
|---|---|---|---|
| real-esrgan | `realesrgan-x4plus` | 4 | photographs, general pictures (the default at 4x) |
| real-esrgan | `realesrgan-x4plus-anime` | 4 | illustration, anime |
| real-esrgan | `realesr-animevideov3` | 2, 3, 4 | video frames, fast (the default at 2x and 3x) |
| waifu2x | `models-cunet` | 1, 2, 4 ... 32 | illustration; `noise` -1 (none) to 3 |
| waifu2x | `models-upconv_7_anime_style_art_rgb` | 2, 4 ... 32 | anime-style art |
| waifu2x | `models-upconv_7_photo` | 2, 4 ... 32 | photographs |
| seedvr2 | `seedvr2-3b` | 1, 2, 3, 4 | video: faces, skin and texture restored, steady over time (the default) |
| seedvr2 | `seedvr2-7b` | 1, 2, 3, 4 | the same, the larger model: finer detail, slower |
| flashvsr | `flashvsr-v1.1` | 2, 3, 4 | video: real footage and faces, the clip as one stream (no seams), fast for a diffusion model |

```json
{"image": "upload:up_1792400000_still.png", "variant": "realesrgan-x4plus", "scale": 4}
{"file": "h3_20261019_121939", "variant": "realesr-animevideov3", "scale": 2}
{"image": "data:image/png;base64,...", "model": "waifu2x", "variant": "models-cunet", "noise": 1, "scale": 2}
{"file": "h3_20261105_134939", "model": "seedvr2", "variant": "seedvr2-3b", "scale": 2, "crf": 16}
{"file": "h3_20261105_134939", "model": "flashvsr", "variant": "flashvsr-v1.1", "scale": 2, "crf": 16, "sparse_ratio": 1.5}
```

- Real-ESRGAN sharpens and invents texture; waifu2x denoises and stays closer to the source. 1x with waifu2x is a
  denoise alone (`noise` 0-3).
- A picture goes through in tiles (`tile`: Real-ESRGAN 200, waifu2x 400) with context around each, exactly as the
  ncnn programs do, so the output is theirs (58-66 dB against them); a larger tile is a little faster and takes more
  memory.
- Transparent pictures keep their alpha (upscaled bicubically, as the programs do).
- SeedVR2 is a one-step diffusion model: it restores as it upscales (natural detail, no waxy skin) and takes a clip
  whole - frames in chunks of 33 together, 4 shared and crossfaded between chunks - so its detail does not flicker
  from frame to frame. It is far slower than Real-ESRGAN (seconds a frame at 2x of 864x480); a picture is a clip of
  one frame.
- FlashVSR v1.1 is a one-step diffusion model too, streamed: 8 frames a step, each attending to the 24 before it, so
  a film of any length goes through as one stream with no seams. Its own keys (at the top level or in `options`):
  `seed` (0), `sparse_ratio` (1.5: the attention's top k; 2.0 a little slower), `local_range` (11; 9 sharper, a
  little less steady), `color_fix` (1: each frame's colour matched to the upscaled input). At 2x of 864 x 480 it takes
  0.16 s a frame on the B70 (about 0.1 with the B65 beside it); 4x runs in tiles. A picture is a still clip.
- With no `model`, a waifu2x variant (`models-...`) picks waifu2x, a `seedvr2-...` one SeedVR2, a `flashvsr-...` one
  FlashVSR and any other Real-ESRGAN.
