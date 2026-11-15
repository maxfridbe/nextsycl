# Budget3D-Idol: a picture to 3D Gaussian splats

Send one picture of one object (`model3d.generate` with `image`). The answer links a PLY of 3D Gaussian splats
(the standard 3DGS layout: positions, colour as degree-0 spherical harmonics, opacity, scales, rotations; y up), and
its settings say the camera the picture was seen from (yaw, pitch, distance, field of view).

## Pictures

- One object, whole, centred. A plain or transparent background works best; an opaque picture is cut out with
  BiRefNet (`remove_background: false` keeps it as it is).
- `options.mode`: `synthetic` (the default: renders and generated pictures, the object framed at 1.2 x its size) or
  `real` (photographs, 1.7 x). `options.crop-margin` sets the framing directly.

## Settings

- `options.budget` (0.0625 to 1, default 1): the share of the 8,192 latent tokens the denoiser runs. Lower is
  faster; the flattest regions, empty space first, are merged, so detail goes where the object has it. 0.5 halves
  the tokens with little loss on most objects; 0.25 and below soften fine detail.
- `options.gaussians` (32,768 to 1,048,576, default 262,144): the splats in the result.
- `steps` (default 50; Turbo 5), `cfg` (default 3; Turbo runs unguided), `seed`.

## Not for

Meshes (`texture`, `faces` and `resolution` do not apply), several views at once, scenes of many objects.
