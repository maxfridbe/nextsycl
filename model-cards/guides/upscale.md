# Upscaling: Real-ESRGAN, waifu2x and SeedVR2

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

```json
{"image": "upload:up_1792400000_still.png", "variant": "realesrgan-x4plus", "scale": 4}
{"file": "h3_20261019_121939", "variant": "realesr-animevideov3", "scale": 2}
{"image": "data:image/png;base64,...", "model": "waifu2x", "variant": "models-cunet", "noise": 1, "scale": 2}
{"file": "h3_20261105_134939", "model": "seedvr2", "variant": "seedvr2-3b", "scale": 2, "crf": 16}
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
- With no `model`, a waifu2x variant (`models-...`) picks waifu2x, a `seedvr2-...` one SeedVR2 and any other
  Real-ESRGAN.
