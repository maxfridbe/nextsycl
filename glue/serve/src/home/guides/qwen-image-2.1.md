# Prompting Qwen-Image 2.1 (pictures)

Qwen-Image 2.1 draws a picture from a description, or edits pictures you give it.

## Generating
- Describe the picture as you would to a painter: subject, its look, the setting, the light, the camera (lens,
  angle, distance), the style ("a photograph", "a watercolor", "flat vector illustration"). Long, concrete
  descriptions work; order the important things first.
- **Text in the picture** renders well: put it in quotes - `a shop sign that reads "OPEN LATE"`.
- `negative_prompt` with `cfg` above 1 steers away from things (the turbo model runs at cfg 1: no negative).
- Sizes: any multiple of 16 up to ~2 megapixels; 1024x1024, 1344x768 (16:9), 768x1344 (9:16) are good defaults.
- Steps: 40 for the full model (`qwen-image-2.1-q8`), 6 for the turbo (`qwen-image-2.1-turbo-q8`); LoRAs that bring
  their own few-step schedule set it themselves.
- Samplers / schedules: ComfyUI's names (`euler` default; `euler_ancestral`, `dpmpp_2m`, `uni_pc` ...); karras and
  exponential schedules suit other kinds of models and do poorly here.

## Editing and composing (`image.edit`, `/v1/images/edits`)
- Give up to 8 pictures; the prompt becomes instructions. The first picture is the one changed, the others are
  references - name them: `put the hat from picture 2 on the fox in picture 1`, `make it night, keep the people`.
- Say what changes and what stays. The output size follows the last picture unless you give one.
