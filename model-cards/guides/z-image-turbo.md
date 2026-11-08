# Z-Image-Turbo

Tongyi-MAI's 6B text-to-image model (Apache-2.0), distilled to 8 steps with no guidance: photographs above all
(people, products, food, street scenes), with English and Chinese text in the picture rendered as written. It reads
Qwen3-4B's reading of the prompt, so long, specific descriptions work better than lists of tags.

## Prompts

- **Describe the picture in sentences**, in English or Chinese: the subject, what it wears or holds, the setting,
  the light and the kind of picture ("a mid-shot phone selfie in a lift mirror", "a film poster", "a close-up product
  photo"). The model's own examples run to a paragraph.
- **Text in the picture:** put each piece in quotes and say where it goes and in what lettering ("the title 'THE
  TASTE OF MEMORY' in large white serif letters across the lower half"). Chinese text works as well as English.
- **No negative prompt and no guidance:** the Turbo is distilled to run without them (cfg 1, or the pipeline's 0); a
  negative prompt or another cfg is refused. Say what you want in the picture rather than what you do not.
- **One seed, one picture:** the Turbo varies little from seed to seed (the model card says so: low diversity), so
  change the prompt rather than the seed to explore.

Example (the model's own): "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral
forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady,
trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night
background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights."

## Sizes

Sides are multiples of 16, from 256 to 2048. 1024 x 1024 is the default; the model's own demo offers these (1024,
1280 and 1536 squared budgets):

| Aspect | 1024 | 1280 | 1536 |
|:---:|:---:|:---:|:---:|
| 1:1 | 1024 x 1024 | 1280 x 1280 | 1536 x 1536 |
| 9:7 / 7:9 | 1152 x 896 | 1440 x 1120 | 1728 x 1344 |
| 4:3 / 3:4 | 1152 x 864 | 1472 x 1104 | 1728 x 1296 |
| 3:2 / 2:3 | 1248 x 832 | 1536 x 1024 | 1872 x 1248 |
| 16:9 / 9:16 | 1280 x 720 | 1536 x 864 | 2048 x 1152 |
| 21:9 / 9:21 | 1344 x 576 | 1680 x 720 | 2016 x 864 |

(the portrait sizes are the landscape ones turned). Below 1024 x 1024 pictures lose detail quickly.

## Steps and shift

**8 steps** (8 model passes: what the README's `num_inference_steps=9` meant on the diffusers of its day). Fewer (4
to 6) are faster and softer; more do not help a distilled model. The scheduler's **shift is 3**; the demo lets it run
from 1 to 10 (higher: more of the steps spent on the composition).
