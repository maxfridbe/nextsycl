# Prompting HunyuanImage 2.1 (pictures)

HunyuanImage 2.1 draws a picture from a description, natively at 2048 x 2048 (2K).

- Describe the picture as you would to a painter: subject, its look, the setting, the light, the camera, the style.
  Long, concrete descriptions in English or Chinese work; order the important things first.
- **Text in the picture**: put it in quotes - `a shop sign that reads "OPEN LATE"`. Quoted text (straight quotes,
  ‘…’ or “…”) also goes through the model's glyph encoder (byT5), which spells it out letter by letter.
- Sizes: sides multiples of 32; 2048x2048 is the model's own, 1024x1024 is ~4x faster and still good, 2560x1536 /
  1536x2560 for 16:9-ish. Very small pictures (below ~768) come out soft.
- **The distilled model** (`hunyuanimage-2.1-distilled-q8`): 8 steps, `cfg` is the embedded guidance (3.25 by
  default; 2.5-4 is the useful range), no negative prompt, one pass a step.
- **The base model** (`hunyuanimage-2.1-q8`): 50 steps, `cfg` 3.5 with a negative prompt (empty by default, as the
  official pipeline) - two passes a step. Its guidance follows the official pipeline: plain CFG for the first steps,
  then adaptive projected guidance (`--opt-guidance cfg` for plain CFG throughout).
- Samplers / schedules: the model's own is `euler` on `shift` (shift 4 distilled, 5 base); ComfyUI's other names work
  (`dpmpp_2m`, `uni_pc` ...) with plain CFG.
- No edits and no LoRAs yet; the official refiner (a second pass at 2K) is not part of this engine.
