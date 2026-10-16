# Pixal3D: one picture to a textured mesh that follows the picture

Pixal3D is TRELLIS.2 with pixel-aligned conditioning: besides the picture's global tokens, every voxel reads the
DINOv3 features at the point where it projects into the picture. The shape therefore keeps the picture's
silhouette and proportions closely; the hidden sides are still invented.

**Give it one picture of one object**, as for TRELLIS.2 (a transparent background is the outline; an opaque one is
cut out by BiRefNet). The object is squared with a tenth to spare. The picture is taken as seen by a camera in
front of the object - set its horizontal field of view when you know it:

| setting | default | what it does |
|---|---|---|
| `--opt-fov DEGREES` | 49.1 (Blender's 35 mm lens) | the picture's horizontal field of view; small values (10-20) for orthographic-looking product shots and renders, 50-70 for phone photos. A wrong value bends the shape (Pixal3D's own pipeline estimates it with MoGe-2, not ported here) |
| `resolution` | 1024 | the cascade (Pixal3D has no 512-only path) |
| `faces`, `texture_size`, `steps`, `cfg`, `seed` | as TRELLIS.2 | |

```sh
nextsycl model3d gen helmet.png --model pixal3d --opt-fov 30 --seed 42 --out helmet.glb
```

The GLB faces the picture along +z, y up. **Quality note:** the shape and texture stages were trained on two
copies of the projected features, the low-resolution DINOv3 patches and the same features upsampled by NAF (a
learned, image-guided upsampler); NAF is not ported, so both halves carry the 16-pixel patch features. Fine detail
that NAF would align to the picture's edges (thin parts, sharp texture boundaries) comes out softer.

**Speed and memory** (a B70, the helmet example): 45 s end to end, 4.6 GiB at the peak (models load stage by stage).
