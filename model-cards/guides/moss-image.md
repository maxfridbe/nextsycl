# MOSS-Image-SFT

OpenMOSS's text-to-image model (5.1B, Apache-2.0): photographs, illustrations, posters and designs, with text in the
picture spelled as quoted. It reads Qwen3.5-9B's reading of the prompt, so it follows long, detailed descriptions.

- **Prompts:** natural language, booru-style tags, or a JSON layout - a short `summary` and a list of `items`, each
  with a description and a bounding box (where text and objects land on the canvas) and, for text, what it says.
  Plain sentences are the everyday way in.
- **Size:** multiples of 16, any aspect ratio, from 0.25 M pixels (512 x 512) to 4 M (2048 x 2048); OpenMOSS
  recommends 1 M, 1.5 M or 2 M. The page computes a size as OpenMOSS's Gradio app does: pick a ratio (1:8 to 8:1) and a
  budget (512², 1M, 1.5M, 2M, 4M) and each side is the budget's root at the ratio, rounded to 16 (16:9 at 1M is
  1360 x 768). 1024 x 1024 is the default.
- **Turbo LoRAs** (`moss-image-turbo-*-lora`, one at a time, strength 1): 4 or 8 steps without guidance (cfg 1), the
  page and the API bring those themselves; 1024 x 1024 in 3 to 5 s.
- **Steps 50, cfg 3** (the model's own). The guidance pass runs the negative prompt (empty unless you give one) with
  block 22 left out, so leave the negative empty unless a picture needs something kept out; fewer steps (12 to 20) are
  fast and rougher.
