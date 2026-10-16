# Making meshes with Hunyuan3D

Hunyuan3D turns one picture of an object into a 3D mesh (GLB). Two models:

- **hunyuan3d-2mini-turbo**: about 5 seconds, a clean shape; no texture.
- **hunyuan3d-2.1**: about 30 seconds on a B70, finer detail (engraved lettering, thin parts); shape only for now.

## The picture
- One object, whole, seen from the front or three-quarters, filling most of the picture. Avoid crops that cut the
  object off, several objects, or scenes: the model builds whatever it sees as one solid.
- A **transparent background** (a PNG with alpha) is the best input: its outline is taken exactly. An opaque
  picture works when its background is one plain colour (it is cut away from the border); a busy background is
  kept and becomes part of the shape - cut it out first.
- Even, soft light and little perspective help; strong shadows and reflections can turn into bumps.
- The back is invented: a picture that shows the object's depth (three-quarter view) gives a better back.

## Settings
- `seed`: another seed, another guess at the hidden sides.
- `steps`: 5 for the turbo (its schedule is built for 5), 50 for 2.1; fewer steps on 2.1 lose detail.
- `cfg`: 5 for both (2mini takes it as its distilled guidance).
- `resolution`: the grid the surface is cut from (380 / 384). Higher keeps finer detail and costs time; 256 is a
  quick draft.
- `faces`: the triangle budget (40,000 by default); the surface is decimated to it. Raise it for close-ups.
- `--opt-tokens N`: the turbo's latent tokens: 3,072 by default; 512 (what Tencent's turbo example samples) is a
  second faster and loses fine detail such as lettering.
