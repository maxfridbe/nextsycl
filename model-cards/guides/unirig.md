# UniRig: a skeleton and skin weights

Give it a mesh (GLB or OBJ); it returns the same mesh - its UVs, textures and vertex order untouched - with a glTF
skeleton and skin weights (four bones a vertex). It rigs characters, animals and objects alike; the bones are named
`bone_0`, `bone_1` ... in the order the model wrote them, the root first.

```sh
nextsycl model3d rig character.glb --model unirig [--seed N] [--opt-beams 15]
```

## Settings

| Setting | Default | What it does |
|---|---|---|
| `--seed N` | random | the point samples and the decoder's sampling: another seed, another skeleton |
| `--opt-beams N` | 15 | the skeleton decoder's beams, sampled (top-k 5, top-p 0.95, temperature 1.5, repetition penalty 3: UniRig's inference); 1: greedy, the same skeleton for the same samples |

## Meshes

- Any triangle mesh, Y up as glTF has it (it is turned Z up for the model, as Blender imports it, and back).
- Over 50,000 triangles, a decimated copy is rigged and its weights carried to every vertex of yours.
- Several separate parts are fine; the weights spread over the mesh's edges, so a part touching nothing takes its
  nearest bones by distance.
- Time: a 7.6K-vertex humanoid takes a few seconds on a B70 (the skeleton's beams, the voxel prior on the CPU, the
  skin model).
