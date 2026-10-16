# AnimateAnyMesh: motion from a sentence

Give it a mesh (GLB or OBJ) and a sentence saying how it moves; it returns the same mesh - its UVs, textures and
vertex order untouched - with 16 frames of motion as glTF morph targets and an animation that plays them (10 frames
a second by default). It does not add a skeleton: the frames are the vertices' own positions.

```sh
nextsycl model3d rig dragon.glb --model animateanymesh --prompt "the dragon flaps its wings and flies up" --seed 7
```

## Writing the prompt

- Say what moves and how, in one short sentence: "a person waving both arms", "the dog wags its tail",
  "a fish swimming, its body swaying left and right", "the object is flying". CLIP reads at most 75 tokens.
- One motion works best; it was trained on clips of 16 frames, so a long sequence of actions will not fit.
- The model works in Blender's frame (Z up): a mesh exported Y up from glTF tools is turned for it and back.
  "Up" in a prompt means the mesh's up.

## Settings

| Setting | Default | What it does |
|---|---|---|
| `--seed N` | random | the motion's starting noise and the first trajectory vertex: another seed, another take |
| `--cfg X` | 3 | guidance toward the prompt (the paper: up to 5 for stronger motion) |
| `--steps N` | 63 | midpoint steps of the flow (two model evaluations each) |
| `--frames N` | 16 | frames in the result: the model makes 16; other counts are resampled linearly in time |
| `--fps X` | 10 | the animation's rate |
| `--opt-traj N` | max(512, vertices / 8) | trajectory tokens: more follow finer parts of the mesh, slower |

## Meshes

- Any triangle mesh; identical vertices (UV seams) are merged for the model and split again in the result.
- A few thousand to some tens of thousands of vertices; the time grows with the vertex count (a 7.6K-vertex
  humanoid: 3.7 s on a B70).
- The motion is relative to the mesh's own shape: thin parts (tails, fingers) far from the rest move most freely.
