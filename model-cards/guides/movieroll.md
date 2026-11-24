# MovieRoll (films up to 60 s at 480p)

Tencent's MovieRoll: HunyuanVideo-1.5 fine-tuned to roll - a silent film made chunk by chunk (16 frames, 0.67 s each),
each new chunk seeing the ones before it, so a minute holds together.

- **Events**: one prompt, or several played one after another - the film is split evenly between them. Give them as
  a JSON list (`["...", "..."]`) or as one text with a line `@@@EVENT_BREAK@@@` between them (upstream's form). A
  new event takes over over about one window (3 s).
- **Write each event in full**: the subject with its full description every time (no "it" across events), what it
  keeps doing (motion that can go on: "walks steadily", not "finally stops"), the setting, and one camera sentence:
  `The camera tracks <subject>, continuously keeping <subject> centered.` or `The camera remains static, keeping
  <subject> centered.` Upstream's prompts are 150-300 words; short ones work, long ones hold better.
- **Length**: `seconds` up to 60 (16n + 1 frames at 24 fps); **size** 864 x 480 or 480 x 864.
- **From a picture** (the I2V card): the picture is the first frame; describe what it shows and how it goes on.
- No sound.
