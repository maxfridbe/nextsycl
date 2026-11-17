# Depth

A depth model reads how far each pixel is: relative inverse depth (near large, far small; no metres). On this box
it makes **depth control videos**: `video.control` with `"control": "depth"` turns a clip into the depth video H3's
Fun ControlNet-Union steers by (`video.generate`'s `control_video`). It is a job in the image queue, as
`video.upscale` is: `wait` (default true), `label`, `reentrant_id`, `image.status`, `image.cancel`.

| model | | |
|---|---|---|
| `depth-anything-v2-small` | Depth Anything V2 Small (DINOv2 ViT-S/14 + DPT), Apache-2.0 | ~25 M parameters, float32 |

```json
{"file": "ref_1792400000_take.webm", "control": "depth", "width": 768, "height": 576, "frames": 124}
```

- The clip is read as for canny: 24 fps from `start_s`, scaled to cover the canvas and cut in the middle, a short
  source's last frame held to `frames`.
- Each frame goes through the model at the size nearest 518 on the 14-pixel grid that keeps its aspect (768 x 576:
  686 x 518), the depth back at the canvas.
- The grey is **one range for the whole clip**: the 0.5th to 99.5th percentile of every frame's values become black
  to white, near bright. A frame's own min / max would make a wall's grey pump as something comes close; the answer's
  `range` says what was used.
- The answer is canny's (`name`, `file`, `width`, `height`, `frames`, `fps`, `seconds`, `source_frames`,
  `source_seconds`, `start_s`, `took_s`) with `model`, `range`, `frame_seconds` and the `job`.
- Only the Small is offered: Depth Anything V2 Base and Large are CC-BY-NC-4.0 (non-commercial).
- `system.version` lists the feature `video.control.depth` only while a depth model is enabled (a client offers depth
  control videos only where they are made); without one a depth request is refused 409 with the call that enables it.
