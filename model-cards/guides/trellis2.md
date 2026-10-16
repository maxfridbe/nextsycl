# TRELLIS.2: one picture to a textured mesh

**Give it one picture of one object.** A transparent background (PNG/WebP with alpha) is used as the object's
outline; an opaque picture is cut out first (BiRefNet, unless `--keep-background`). The object is cropped, squared
and centred, so framing does not matter - but it should be the whole object, seen clearly, ideally from the front
or three-quarters, on a plain background. Product shots, renders, figurines, props and characters work best;
scenes, several objects or heavy occlusion do not.

**What you get:** a GLB with a base colour texture and a metallic-roughness texture (glTF PBR), y up, about a unit
across. The back and the hidden sides are invented by the model.

| setting | default | what it does |
|---|---|---|
| `resolution` | 1024 (`trellis2`), 512 (`trellis2-512`) | 512: one shape pass, coarser, fast. 1024: the cascade - the 512 shape refined at 1024 (a token budget of 49152; denser objects step the resolution down by 128) |
| `faces` | 300000 | the triangle budget (the surface is clustered then decimated to it) |
| `texture_size` | 2048 (1024 at 512) | the side of both textures in texels |
| `steps` | 12 | each stage's Euler steps |
| `cfg` | 7.5 | the guidance of the structure and shape stages (the texture stage runs unguided, as the model was trained) |
| `seed` | random | the same seed and picture give the same mesh |
| `--opt-max-tokens N` | 49152 | the cascade's budget for the second shape pass |
| `--opt-remesh` | off | remesh the surface evenly before decimating (slower) |

```sh
nextsycl model3d gen helmet.png --model trellis2 --seed 42 --out helmet.glb
nextsycl model3d gen goblin.jpg --model trellis2-512 --faces 150000
```

**Speed and memory** (a B70; the helmet example, 4267 structure voxels, 20323 in the cascade's second pass): 512
in 21 s, the 1024 cascade in 80 s (the second shape pass 29 s, texture 18 s, decoders 11 s, mesh and bake 9 s); the
models load stage by stage, so the card needs about 9 GiB at the 1024 peak (the decoders at 8 million voxels), 3 GiB
at 512. `--opt-resident` keeps every model loaded between requests (faster repeats, ~11 GiB more).
