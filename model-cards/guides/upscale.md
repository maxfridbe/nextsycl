# Upscaling: Real-ESRGAN and waifu2x

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

```json
{"image": "upload:up_1792400000_still.png", "variant": "realesrgan-x4plus", "scale": 4}
{"file": "h3_20261019_121939", "variant": "realesr-animevideov3", "scale": 2}
{"image": "data:image/png;base64,...", "model": "waifu2x", "variant": "models-cunet", "noise": 1, "scale": 2}
```

- Real-ESRGAN sharpens and invents texture; waifu2x denoises and stays closer to the source. 1x with waifu2x is a
  denoise alone (`noise` 0-3).
- A picture goes through in tiles (`tile`: Real-ESRGAN 200, waifu2x 400) with context around each, exactly as the
  ncnn programs do, so the output is theirs (58-66 dB against them); a larger tile is a little faster and takes more
  memory.
- Transparent pictures keep their alpha (upscaled bicubically, as the programs do).
- With no `model`, a waifu2x variant (`models-...`) picks waifu2x and any other picks Real-ESRGAN.
