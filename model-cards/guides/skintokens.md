# SkinTokens (TokenRig): a skeleton and skin weights as one token sequence

Give it a mesh (GLB or OBJ); it returns the same mesh - its UVs, textures and vertex order untouched - with a glTF
skeleton and skin weights (four bones a vertex). One decoder writes the skeleton and then, for every joint, four
"skin tokens" (codes of a learned skin codebook) that a small VAE decoder turns into that joint's weights. The bones are
named `bone_0`, `bone_1` ... in the order the model wrote them, the root first.

```sh
nextsycl models pull skintokens
nextsycl model3d rig character.glb --model skintokens [--seed N] [--opt-beams 10] [--opt-skin-end upstream|fix]
```

## Settings

| Setting | Default | What it does |
|---|---|---|
| `--seed N` | random | the point samples and the decoder's sampling: another seed, another rig |
| `--opt-beams N` | 10 | the decoder's beams, sampled (top-k 5, top-p 0.95, temperature 1, repetition penalty 2: the demo's); 1: greedy, the same rig for the same samples |
| `--opt-skin-end upstream\|fix` | upstream | where the skin tokens end (below) |

## Files

`tokenrig` (grpo_1400.ckpt, 1.1 GB) is all the engine reads: it carries the skin VAE too (a bfloat16-rounded copy of the
standalone `skinvae` checkpoint, last.ckpt, 487 MB, which is optional and unused at inference). Both are Lightning
checkpoints, read as published without Python.

## What follows the demo, and what does not

- The sequence is `bos, articulation class, the skeleton, its end, four skin tokens a joint, the model's end`. The demo
  forces the model's end token one token early, so the **last joint's fourth skin token is the end token** and reads as
  code 0 instead of what the model would have said (its logits there clearly prefer a skin token). The default
  reproduces that exactly, as the checkpoint was evaluated; `--opt-skin-end fix` lets the model say that token.
- The demo's `--use_postprocess` (merging and pruning bones) and `--use_skeleton` (a given skeleton) are not ported:
  the rig is what the decoder writes, the mesh's own vertices get the weights.
- Weights reach the mesh's vertices as the demo's `Asset.from_data` does: the 8 nearest of the 54,000 surface samples,
  blended by 1 / (distance + 1e-8); then the four largest a vertex, renormalized.

## Meshes

- Any triangle mesh, Y up as glTF has it (it is turned Z up for the model, as Blender imports it, and back). No welding
  or decimation: the samples are 16,384 of your vertices and the rest on the surface, whatever the mesh's size.
- It writes the "articulation" class of skeletons (the class token the demo asks for); characters and creatures suit it
  best, other shapes get a rig, not always a good one.
- License: MIT (code and weights).
