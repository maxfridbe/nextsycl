## Background removal (image.remove_background): BiRefNet and IS-Net general-use

`image.remove_background` is nextcuda's method with nextcuda's request and answer (the platform calls both boxes):
`image` (data URL, base64 or `upload:<name>`), `model` (`birefnet` | `isnet-general-use`; none: the loaded remover,
else the first enabled), `mask`, `post_process_mask`, `response_format`, the job options. The answer:
`{created, data: [{b64_json | url, width, height, model, mask_b64_json | mask_url}], nextsycl: {seconds}}` - the RGBA
picture at the picture's size, its own RGB with the mask as straight alpha (rembg's naive_cutout premultiplies; this
does not), the mask fields only with `mask: true`. A model that is not a remover is `invalid_params`, "<id> does not
remove backgrounds (birefnet, isnet-general-use do)", refused at submit (wait false too).

The engine is `image/rembg` (kernels `kernels/image/rembg`), after nextcuda's:

- `birefnet`: `crates/remover` as the 3D engines use it (BiRefNet Swin-L in float32: the shared vision kernels and the
  diffusion kernels' GEMMs) - PIL's bilinear to 1024, ImageNet normalization, the last prediction's sigmoid as
  ToPILImage's 8 bits, PIL's bicubic back. Its file is trellis2's (`model-cards/model3d/trellis2.yml`, key `birefnet`).
- `isnet`: IS-Net general-use as rembg runs it - rembg's ONNX file read by a small protobuf reader (`onnx.rs`: the
  graph's convolutions by their module path, batch norms already folded; no ONNX runtime), the RSU U-net in float32
  (the shared vision im2col and oneDNN's GEMM; its own ceil-mode max-pool, half-pixel bilinear upsampling, bias + ReLU
  and sigmoid, compiled precise), rembg's preparation (Pillow's Lanczos to 1024, / the largest value in float64,
  - 0.5) and mask (the prediction scaled to its own [min, max] in float32, x 255 truncated, Lanczos back).
- `post_process_mask`, either model: rembg's `post_process` - `morphologyEx(MORPH_OPEN)` with the 3 x 3 ellipse (a
  cross; outside the picture takes no part), `GaussianBlur((5, 5), 2)` with BORDER_REFLECT_101 in OpenCV's 8-bit fixed
  point (the 1-D kernel 8 fraction bits, rows then columns, rounded once), 0 below 127 else 255. Its unit tests hold
  OpenCV 5.0's own outputs (`image/rembg/src/lib.rs`).

### The reference

`reference/rembg/ref.py` on the CPU in float32, every stage dumped:

- isnet: rembg 2.0.67 through its own session (`new_session("isnet-general-use")`, onnxruntime 1.31.1's CPU
  provider; `normalize`, `inner_session.run`, `predict`, `bg.post_process`, `remove(..., post_process_mask=True)`).
- birefnet: ZhengPeng7/BiRefNet's own code at the card's revision (e2bf8e44; transformers 5.22, torch 2.15 CPU) as
  TRELLIS.2's rembg.BiRefNet calls it, then rembg's post_process (opencv-python-headless 5.0.0.93).
- **Pillow 12.2.0 in both.** Pillow 13.0 resizes the vertical pass first when the height shrinks more than twice the
  width does (Resample.c's new `horizontal_first`); with 13.0 the IS-Net mask of a 1000 x 667 photograph moved by up
  to 3 levels in 0.8% of its pixels, and a portrait's (987 x 1481) Lanczos input by 4.7e-2. nextsycl's `Picture` and
  nextcuda resize across first, as Pillow up to 12.2 does, so both boxes give the same bytes; a platform on Pillow 13
  will see its rembg drift from both by those levels.

The pictures: rembg's own examples at v2.0.67 (commit 066d88ba, `examples/`): animal-1 (1000 x 667), anime-girl-1
(850 x 601), car-1 (480 x 360), food-1 (6000 x 4000), girl-1 (600 x 399), plants-1 (987 x 1481, a PNG with alpha:
its RGB).

### Checked (the B65, `--gpu 1`)

`nextsycl image remove-background --check DUMPS --file MODEL --gpu 1`: the preparation against input.npy, the
network fed the reference's own input against its prediction, the mask the whole way and post_process_mask of it
against the reference's bytes.

IS-Net general-use (nextcuda's bar: the prediction at most 6.6e-7 relative, the mask 99.998-100% equal and never more
than 1 level off, the post-processed mask 100% equal):

| picture | input | prediction: max, relative | mask: equal, pixels off, max | post-processed: equal, pixels off |
|---|---|---|---|---|
| animal-1 | exact | 2.65e-6, 2.0e-7 | 99.9993%, 5, 1 | 100%, 0 |
| anime-girl-1 | exact | 2.80e-6, 2.8e-7 | 99.9984%, 8, 1 | 100%, 0 |
| car-1 | exact | 1.58e-6, 1.4e-7 | 100%, 0, 0 | 100%, 0 |
| food-1 | exact | 8.94e-7, 8.7e-8 | 99.9995%, 131, 1 | 100%, 0 |
| girl-1 | exact | 2.38e-6, 1.8e-7 | 99.9996%, 1, 1 | 100%, 0 |
| plants-1 | exact | 5.07e-6, 6.3e-7 | 99.9969%, 45, 1 | 99.9999% or 100%: 1 or 0 |

The prediction differs from onnxruntime's only by float32 summation order (oneDNN's GEMM against MLAS's), and
oneDNN's order is not the same from run to run on the card: a mask pixel at a truncation boundary flips by one level
in a few runs and not others (plants-1's mask 45-63 pixels off; its post-processed mask 1 pixel off in two runs of
the check, 0 through the server). With the prediction rembg itself made (pred.npy), the mask and post-processed mask
are byte for byte rembg's on all six: the preparation, scaling, Lanczos and post-processing are exact; what is left
is the float32 noise of the network.

BiRefNet (against its own PyTorch code):

| picture | input | logits: max, relative | mask: equal, pixels off, max | post-processed: equal, pixels off |
|---|---|---|---|---|
| animal-1 | exact | 8.62e-4, 1.0e-6 | 99.9990%, 7, 1 | 100%, 0 |
| anime-girl-1 | exact | 2.84e-4, 9.3e-7 | 99.9986%, 7, 1 | 100%, 0 |
| car-1 | exact | 2.41e-3, 4.0e-6 | 99.9965%, 6, 1 | 100%, 0 |
| food-1 | exact | 3.80e-3, 2.3e-6 | 99.9987%, 304, 1 | 99.99999%, 2 of 24 M |
| girl-1 | exact | 2.61e-4, 7.7e-7 | 99.9983%, 4, 1 | 100%, 0 |
| plants-1 | exact | 3.36e-4, 1.4e-6 | 99.9956%, 65, 1 | 100%, 0 |

(BiRefNet's logits are large - tens - so their maximum difference is large in absolute terms; the mask, its sigmoid in
8 bits, is never more than one level off.)

Through the image server too (`POST /v1/images/remove-background` on a slot-37 server of the build copy, the six
pictures as PNG data URLs, mask true, with and without post_process_mask): every answer RGBA and grey at the picture's
size, width and height right, RGB equal to the picture's and A equal to the mask byte for byte, the masks as in the
tables; `response_format: url` links `<name>-cutout.png` and `<name>-cutout-mask.png`.

### Speed (the B65, the image server's `nextsycl.seconds`: decode not counted, the preparation, network and mask)

| model | 0.2-1 MP pictures | food-1, 24 MP | nextcuda (RTX 3060) |
|---|---|---|---|
| isnet-general-use | 0.21-0.24 s (the network 0.19-0.20 s) | 0.50 s; with post_process_mask 0.70 s | ~0.2 s |
| birefnet | 0.65-0.68 s (the network 0.63 s) | 0.82 s; 1.0 s | ~1.5 s |

The first request after a load is slower (oneDNN makes its primitives: isnet 0.33-0.40 s, birefnet 1.15 s). On the
24 MP picture the CPU's share grows: the Lanczos passes and post_process_mask (0.18 s) in Rust. Not tuned further:
IS-Net is at nextcuda's speed, BiRefNet at twice it.
