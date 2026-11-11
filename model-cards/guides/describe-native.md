# An image model's text encoder as a describer

This is not a model you prompt: it is an image model's own text encoder - the language model that reads that image
model's prompts - loaded whole (its head and vision tower too) so it can look at a picture and write the prompt that
would draw it. `image.describe` runs it with `describer: native` (or `describer: <this id>`); the instruction is the
same as every describer's, written for the target's family (docs/api/image-describe.md).

- It sees one picture at a time, fitted to the request's `max_pixels` (default ~1 MP) on a 32-pixel grid.
- It writes without thinking and keeps no prompt cache: every describe reads the instruction and the picture anew.
- It is a smaller language model than the default describer (`best`, a 27B chat model): expect shorter, plainer and
  sometimes wrong prompts - but in the words the image model's own encoder knows.
- Its files are its image model's: registering it (`system.models.pull`) downloads nothing.
- As a reading model it also answers `/v1/ocr` on its image server: the picture, an instruction as `prompt`, a system
  turn as `system`.
