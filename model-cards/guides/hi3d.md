# Refining meshes with Hi3D

Hi3D v1.0 takes a **coarse mesh** and a **picture of the same object** and returns a high-resolution, geometry-only
mesh (GLB: no texture, recentred, scaled to radius 1). It keeps the coarse mesh's structure and adds the detail the
picture shows; what the picture hides is a plausible completion.

## What to give it
- `image`: one picture of the object, as for a generator: the whole object, a plain or transparent background (a
  transparent PNG is framed by its alpha; an opaque one is cut out by the background remover unless
  `remove_background: false`).
- `mesh` (a GLB or OBJ) or `file` (a GLB made here): the coarse mesh. It must be the object in the picture, posed
  the same way. Without one, model3d.refine first makes it with a generator (hunyuan3d-2mini-turbo by default:
  seconds) from the same picture. Severe errors in the coarse mesh (missing parts, wrong topology) stay.

## Settings
- `resolution`: 512 (one pass), 1024 (two passes, the default) or 1536 (three; slow and memory-hungry: only on a
  free 32 GB card). hi3d-1.0-512 does 512 only.
- `steps`: scheduler updates a pass (25). Fewer is faster and coarser.
- `faces`: the final triangle budget (1,000,000 at 512 and 1024, 2,000,000 at 1536).
- `seed`: another seed, another guess at hidden detail.
- `options.euler`: first-order updates (half the transformer calls, a little softer); the default is Heun.

There is no prompt and no guidance setting: guidance follows the model's own schedule.

## Licence
Hi3D Open License Agreement v1.0: commercial use is allowed; a product above 200,000 monthly users or USD 1M monthly
revenue must ask the licensor for a separate licence; commercial integrations show "Powered by Hi3D 1.0". Meshes it
makes carry no duties of their own.
