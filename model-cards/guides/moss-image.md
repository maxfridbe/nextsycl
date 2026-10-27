# MOSS-Image-SFT

OpenMOSS's text-to-image model (5.1B, Apache-2.0): photographs, illustrations, posters and designs, with text in the
picture spelled as quoted. It reads Qwen3.5-9B's reading of the prompt, so it follows long, detailed descriptions.

- **Prompts:** natural language, booru-style tags, or a JSON layout - a short `summary` and a list of `items`, each
  with a description and a bounding box (where text and objects land on the canvas) and, for text, what it says.
  Plain sentences are the everyday way in.
- **Size:** multiples of 16 (1024 x 1024 is the default); other aspect ratios work at the same area.
- **Turbo LoRAs** (`moss-image-turbo-*-lora`, one at a time, strength 1): 4 or 8 steps without guidance (cfg 1), the
  page and the API bring those themselves; 1024 x 1024 in 3 to 5 s.
- **Steps 50, cfg 3** (the model's own). The guidance pass runs the negative prompt (empty unless you give one) with
  block 22 left out, so leave the negative empty unless a picture needs something kept out; fewer steps (12 to 20) are
  fast and rougher.
