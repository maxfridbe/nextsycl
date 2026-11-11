# MiniMax-H3-Image

MiniMax's image model on the H3 denoiser: text to image, and edits guided by pictures, in **6 steps** with no guidance
(no negative prompt, cfg 1), from 1 to **16 megapixels**. It is distilled for 2048 x 2048 (the default). nextsycl renders
prompts **as written**: MiniMax's prompt enhancer (a Qwen3.8-27B fine-tune) is not run, so write the full prompt
yourself.

## Prompts

- **Describe one scene** plainly: the subject, what it is doing, the setting, the light and where it comes from, the
  kind of picture ("35mm photo", "matte painting", "Swiss design poster").
- **Lettering**: put every word in quotes and say where it sits - `a chalkboard by the door reading "OPEN LATE"`.
  Short strings work best; Q3_K_M keeps lettering right at the default size, larger files hold small text better.
- Examples that work (the GGUF card's): `A night market food stall in Taipei, a hand-painted menu board reading "BEEF
  NOODLES 120" and "SCALLION PANCAKE 60", steam rising from a pot, warm lantern light, 35mm photo`;
  `A minimalist poster for a jazz night, a saxophone made of brass leaves on a cream background, the words "HEARTH JAZZ"
  in bold letters at the top and "FRIDAY 9 PM" at the bottom, Swiss design`.

## Edits

Give the picture to change (and up to 15 more as references) and say what to change: `Give him a red knitted beanie.
Keep everything else exactly the same.` The prompt can name the pictures as `<Picture 1>`, `<Picture 2>` ...; each is
shown to the text encoder as "Picture i: " and its vision tokens, and to the denoiser as its own latents. Without a
size the canvas is picture 1's, its short edge brought into 1,024 .. 2,048. No masks. A strength below 1 (or
`options.start-sigma`) also starts the denoise from picture 1's latents instead of noise; with `reference: false` the
picture is only that start (an img2img: the prompt alone guides it).

## Sizes

Sides are multiples of 32, from 256 to 16 MP. The ratio and budget presets give 1, 4, 8 and 16 MP at 1:1, 4:3, 3:2,
16:9 and their transposes (2048 x 2048, 2912 x 2912, 4096 x 4096, 5024 x 3360 ...). Time grows faster than the area
(attention): see docs/ports/minimax-h3-image.md for the measured times.

## Steps

6 by default: the model's six output heads were distilled on its own grid (1, 0.968, 0.888, 0.75, 0.533, 0.232, 0).
Other step counts read that grid at even positions and blend the heads by the share of the grid each step spans
(fewer steps are faster and softer); `options.sigmas` takes a grid of your own.

## 2 steps (the pDMD LoRA)

`minimax-h3-image-2step` adds the pDMD 2-NFE LoRA (Apache-2.0; pdmd2026, ComfyUI layout) at strength 1 and samples **2
steps** on ComfyUI's shift-12 schedule (sigmas 1, 0.923, 0) - a third of the denoiser's work. Its recipe, from the LoRA's
card: no guidance, prompts long and complete; if 2 steps drifts on a prompt, ask for 3 or 4 (`steps`). The LoRA alone
(`minimax-h3-image-pdmd-2step-lora`) rides on any of the cards above as a request LoRA (`loras`) and brings the same
2-step / shift-12 preset; any other strength scales the distillation.

- **Detail pass**: render at half size, then `image.edit` the result at the final size with `reference: false`,
  `strength: 0.5` and `options: {"start-sigma": 0.5}`, `steps: 3` - the demo Space's grid (0.5, 0.471, 0.367, 0; the
  start sigma wins over the strength, which only has to be below 1 for an img2img); 0.4 - 0.55 polishes, 0.7 adds detail
  but may drift or speckle flat areas. A `strength` without start-sigma is ComfyUI's denoise instead: under shift 12,
  strength 0.5 at 3 steps starts at sigma 0.923, not 0.5.
- **Edits** take the picture as a reference at 2 steps (as the base model's); to stay closer to the source, add a start
  sigma of 0.55 or less (`options.start-sigma`): the denoise then starts from the picture's own latents.
- **Repeats are cheap**: the card keeps the text encoder's result for the last 8 prompts and the VAE latents of the
  last 8 pictures, so another seed, size or polish of the same prompt, or another edit of the same picture, costs only
  the denoise and the decode. A changed prompt runs the 32B text encoder again.

## Files

| Model | Denoiser | On the card | Notes |
|---|---|---|---|
| `minimax-h3-image` | Q4_K_M GGUF | 10.1 GiB | the GGUF maker's recommendation |
| `minimax-h3-image-q5` | Q5_K_M GGUF | 12.3 GiB | closer to bf16 |
| `minimax-h3-image-q8` | Q8_0 GGUF | 19.1 GiB | the closest to bf16 of the GGUFs, and the fastest of them up to 4 MP (int8 products) |
| `minimax-h3-image-q3` | Q3_K_M GGUF | 7.7 GiB | the smallest download and the smallest on the card; the most error |
| `minimax-h3-image-int8` | ComfyUI int8 ConvRot | 18.0 GiB | int8 products: ~30 % faster than the GGUFs; about as close as Q8_0 |
| `minimax-h3-image-2step` | int8 ConvRot + pDMD LoRA | 18.0 + 1.8 GiB | 2 steps; the text encoder's and pictures' results cached: the live editor's card |

All share H3's Qwen3-VL 32B text encoder (Q4_K_M, streamed from pinned host memory: ~14 GiB of RAM) and the image
VAE (0.75 GiB on the card).

## License

The MiniMax-H3-Image Community License: not licensed in the EU, the UK, South Korea or the US; MiniMax's written
authorization is needed before serving it to third parties as a hosted service or above US$1 million yearly revenue;
outputs may not be used to improve another model. Read the LICENSE that comes with the files.
