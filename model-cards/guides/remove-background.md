# Background removal

`image.remove_background` keeps a picture's object and makes its background transparent: an RGBA PNG at the
picture's size (its own colours, the mask as straight alpha) and, with `"mask": true`, the 8-bit mask on its own.
A job like the other image methods (wait, reentrant_id, label, image.status).

| model | | |
|---|---|---|
| `birefnet` | BiRefNet (Swin-L), MIT | the best edges: hair, fur, thin parts |
| `isnet-general-use` | IS-Net general-use, rembg's default model | what rembg gives today |

```json
{"image": "upload:up_1792400000_chair.png", "model": "birefnet", "mask": true, "post_process_mask": true, "response_format": "url"}
```

- `image`: PNG, JPEG or WebP as base64, a data: URL or `upload:<name>` (image.upload).
- `post_process_mask`: rembg's post-processing - an opening, a blur and a threshold at 127: a hard 0 / 255 mask
  without specks or soft edges. Either model.
- `response_format`: `b64_json` (default) or `url`.
- The answer's `data[0]`: `b64_json` or `url` (the RGBA picture), `width`, `height`, `model`, and with `mask`
  `mask_b64_json` or `mask_url`.
- rembg's own cut-out multiplies the colours by the mask; here they are kept (multiply RGB by A / 255 for rembg's bytes).
