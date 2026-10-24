# Ming-Image-0.1-Design

A design model: UI screens, landing pages, posters, infographics, slides, logos, stickers - compositions with a lot of
text in them, and that text spelled right. Photographs and painterly scenes are not what it is for.

- **Say what is on the canvas, and where.** Layout first (a header, two columns, a card grid), then each block's
  words in quotes - the words you quote are the words it writes. It was trained on Figma-style JSON captions
  (elements with their text, font, colour and position) and takes them as the prompt too; plain sentences work.
- **Transparent pictures:** ask for RGBA (`rgba: true`, `--rgba`): the engine puts the model's phrase
  `RGBA, 4-channel, transparent background` before the prompt (a prompt that starts with one of the README's ten
  fixed phrases keeps its own), and the picture keeps the VAE's fourth channel. With these GGUF files the alpha comes
  back opaque (240-255) - as it did through ComfyUI on the same files; the picture is the subject on a plain
  background, ready for a cut-out.
- **Size:** the model works at 1024 x 1024 (default) or 2048 x 2048 - other sizes are made at the nearest of its
  aspect ratios at that scale and resized to the size asked. 2048 is the recommended quality and about four times
  the time.
- **Steps 12, cfg 1** (the model's own). Above 1, guidance runs a second pass against an empty condition: the model
  has no negative prompt, so one given is not used.
