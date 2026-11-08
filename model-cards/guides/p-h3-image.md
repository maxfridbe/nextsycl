# P-H3-Image

PrunaAI's image model on MiniMax H3's denoiser: text to image, and edits with up to 8 pictures, in 8 steps with no
guidance (no negative prompt, cfg 1). It renders prompts **as written** here: nextsycl does not run Pruna's prompt
upsampler (Qwen3.5-9B), so write the full prompt yourself - the model is trained on long, specific captions and gives
weak results from one-line requests.

## Prompts

Four things matter most (Pruna's guidance):

1. **Say what the image is.** Open with one sentence naming the real object the picture is: "a photograph by an
   experienced street photographer", "a frame from a Japanese anime TV series", "a two-color risograph print", "a flat
   scan of a handwritten recipe card". This sets the look more reliably than any style word.
2. **Describe one scene.** A viewpoint, one main subject caught in a moment, a few supporting elements, and light with
   a source and a direction.
3. **Spell out every word.** Text without its exact words turns into pseudo-text. Put each piece of text in quotes,
   keep each string short, say where it sits, and end with "no other text". If you want no text, do not mention signs
   or labels at all.
4. **Leave out hype words.** Stunning, epic, cinematic, vibrant, hyper-detailed, 8k, masterpiece push it toward a
   glossy, oversaturated, airbrushed look.

Styles that work especially well: documentary and editorial photography, portraits with real skin texture, film
frames, TV-frame anime, ink and charcoal drawings, screen-printed posters, flat graphic design with short quoted text.

Example (the model's own): "A German Expressionist oil painting of a Berlin street at night in the manner of Ernst
Ludwig Kirchner, with gaunt figures hurrying past a glowing gas lamp."

## Sizes

Sides are multiples of 32. It is at its best at **8 MP**; 4 MP and lower work, with more artifacts in faces, hands and
small details. The default is 2912 x 2912. The page's ratio and budget presets (1, 4, 8 MP) give Pruna's sizes:

| Aspect | 8 MP | 4 MP (edits) |
|:---:|:---:|:---:|
| 1:1 | 2912 x 2912 | 2048 x 2048 |
| 4:3 | 3360 x 2496 | 2368 x 1760 |
| 3:4 | 2496 x 3360 | 1760 x 2368 |
| 3:2 | 3552 x 2368 | 2496 x 1664 |
| 2:3 | 2368 x 3552 | 1664 x 2496 |
| 16:9 | 3872 x 2176 | 2720 x 1536 |
| 9:16 | 2176 x 3872 | 1536 x 2720 |

The composition changes with the size (the noise does).

## Edits (image.edit)

Picture 1 is the one edited; pictures 2 to 8 are references the prompt names as Picture 2, Picture 3 ... Without a
size the edit is made at picture 1's size, scaled up to at least 4 MP (at most 8 MP); the other pictures keep theirs
within 1 to 8 MP. A size changes the canvas, not the pictures. Edits read the pictures as references only: no
strength below 1 (no img2img), no masks.

Write the edit as the finished result and say what stays, in this form (an instruction that does not start with
"Edit <Picture 1>." gets that prefix):

```txt
Edit <Picture 1>. <The change, described as the finished result>. Keep <everything that must stay> exactly as it is.
```

- Point to the exact thing ("the old man's faded blue cap", not "his hat") and where it is.
- Describe the result, not the action; give new things a size next to something already there.
- Say what must not change (faces, poses, the background, the light) - without that list it tends to redraw the
  whole picture.
- Quote new text exactly and name the text around it that must stay.

## Limits

Hands can break and small people in the background can come out distorted (8 MP helps); small or dense text turns into
pseudo-text. The license (MiniMax H3 Community License) excludes the European Union, the United Kingdom, South Korea
and the United States without a separate license from MiniMax, and products built on it must show "Powered by MiniMax
H3"; the text encoder's Qwen3-VL-4B parts are Apache-2.0.
