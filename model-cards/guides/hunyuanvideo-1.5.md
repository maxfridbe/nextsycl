# HunyuanVideo-1.5 (480p text to video)

Tencent's 8.3B video DiT: a silent 5-second clip (121 frames at 24 fps) at 480p from a prompt.

- **Describe the video, in full sentences**: the subject and what it does, then the setting, the light and style, and
  the camera ("the camera tracks slowly beside it"). The text encoder's template asks for the main content, the
  objects' colours, shapes and positions, actions and movement, the background and light, and the camera; a prompt
  that covers those reads best. English or Chinese.
- **Text in the picture**: put it in quotes - `a neon sign that reads "OPEN LATE"` - and the glyph encoder spells it.
- **Size**: the clip takes the 480p frame nearest the width and height asked in shape (16:9 is 848 x 480, 9:16 480 x
  848, 1:1 640 x 640). `seconds` up to 5 (121 frames) is what it was trained on; frames are 4k + 1.
- **Steps**: the cfg-distilled card defaults to its fast take - 30 steps with DeepCache and Sage attention, the 5 s
  clip in about 3 minutes on a B70 (5 on a B65). It was trained for 50: `steps: 50` with the card's `NS_HV15_CACHE`
  / `NS_HV15_ATTN` emptied gives the exact 50-step take (11 minutes). The base card runs CFG 6, twice the passes a
  step, 50 steps.
- **Faster takes** (measured on one prompt; docs/ports/hunyuanvideo15.md): `steps` 30 gives a different but sharp clip
  of the same scene from the same seed (1.6x); the engine's options `cache: deepcache` (upstream's step caching,
  1.9x the steps) and `attention: sage` (int8 attention, 1.3x) keep the same clip by eye with small changes in detail.
  All three together: the 5 s clip in about 3 minutes on a B70 instead of 11. The two options are the card's
  `NS_HV15_CACHE` / `NS_HV15_ATTN` (`--opt-cache`, `--opt-attention`) or a daemon job's `cache` / `attention`;
  `video.generate` carries `steps` only.
- No sound, no keyframes, references or ControlNet: those are MiniMax H3's (`minimax-h3`).
