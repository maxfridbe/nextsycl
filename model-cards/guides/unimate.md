# UniMate: a rigged mesh set in motion by a sentence

Give it a mesh that already has a skeleton and skin weights (a GLB from UniRig or RigNet, or any GLB with a skin) and
a sentence; it returns the same mesh with a skeletal animation named "skeleton": 60 frames at 30 fps (2 s), a
rotation for every joint on every frame and a path for the root. Nothing about the mesh changes - vertices, UVs,
textures, weights - so it plays in any glTF viewer or engine.

```sh
nextsycl model3d rig humanoid.obj --model unirig --out humanoid-rigged.glb
nextsycl model3d rig humanoid-rigged.glb --model unimate --prompt "A person walks forward." --seed 1 --out walk.glb
```

**Licence.** The weights are CC BY-NC 4.0: research and other non-commercial use only. The code is MIT.

## Joint names matter, and generic names are the weak spot

UniMate reads each joint as a text label ("Left Front Thigh") through flan-T5, plus the skeleton's graph (parents,
distances, depths, spectral position) and the rest pose. The labels are the model's way of knowing which joint is a
leg. The engine cleans the names by rules (no LLM): `mixamorig:LeftUpLeg` becomes "Left Thigh", `FR_thigh` "Right Front
Thigh", `Bip01 R Thigh` "Right Thigh".

**The joints UniRig and RigNet write are named `bone_0`, `bone_1`, ... and carry no meaning.** Such a name is labelled
"Bone"; the model then has only the graph and the rest pose to go by. Measured on the test chain (UniRig's skeleton of
`reference/meshes/humanoid.obj`, 28 joints, "A person walks forward.", B70):

| | seed 1 | seed 2 |
|---|---|---|
| motion amplitude (rms joint rotation from rest) | 11 deg | 25 deg |
| root travel in 2 s | 3.5 | 0.3 |

The rig is walking in one take and mostly standing in the other; the rotation between the two takes is 24 deg rms,
twice the amplitude of the first. The same prompt on a rig with real names (the Unitree Go2 example from the UniMate
repository, "A robot dog walks forward.") is stable: amplitude 50 deg in both seeds, root travel 3.0 and 2.9, the
takes 18 deg rms apart (0.36 of the amplitude). So for a UniRig or RigNet skeleton expect a weaker, less repeatable
motion than the paper shows, and try several seeds. Renaming the joints in the GLB before animating (Mixamo or
`Left Thigh` style names) is the real fix; nothing in this engine invents the names from the geometry.

### `--opt-names structure`

Leaves every name out: the model reads the graph alone. Same rig, same seed, same prompt, compared with the default:

| rig | rotation difference (rms) | amplitude, default | amplitude, structure | root travel, default | root travel, structure |
|---|---|---|---|---|---|
| Go2 (real names) | 75 deg | 50 deg | 84 deg | 3.0 | 17.1 |
| UniRig humanoid (`bone_N`) | 61 deg | 11 deg | 64 deg | 3.5 | 14.0 |

Without names the motion is not a quieter version of the same walk, it is a different and exaggerated one: joints
swing 60-85 deg and the root runs several times too far (the numbers are in the rig's units, a body is about 2 across).
The model never saw a skeleton without labels, so this mode is out of distribution. It exists to show what the names
contribute and as a fallback when a name would mislead (a rig whose `Arm` is a wing); use it knowing the result is
rougher. On a rig with no recognizable names, the default ("Bone" everywhere) was gentler than structure-only.

## Writing the prompt

- One short sentence in the style of the training captions (HumanML3D): "A person walks forward.", "A person jumps.",
  "A dog runs.". Describe what the whole body does; there is no per-joint control.
- The root moves along the way the rig faces. Facing is read from the left/right names, else from the rig's mirror
  symmetry and its feet, else +Z; `--opt-facing +z|-z|+x|-x` overrides it when a walk goes sideways or backwards.
- flan-T5's tokenizer is run without the Unicode (NFKC) normalization table: write the prompt in plain ASCII.

## Settings

| Setting | Default | What it does |
|---|---|---|
| `--seed N` | random | the starting noise: another seed, another take |
| `--cfg X` | 3 | guidance toward the prompt (1 or less: no guidance, half the work) |
| `--frames N` | 60 | frames in the result; the model makes 60, other counts are resampled in time |
| `--fps X` | 30 | the animation's rate |
| `--opt-facing` | auto | which way the rig looks |
| `--opt-names` | auto | `structure` leaves the joint names out (above) |

`--steps` does nothing: the sampler is adaptive (about 100-130 evaluations, 12-15 s on a B70 whatever the mesh size,
the skeleton being the only input).

## Limits

- At most 71 joints and a single root; more joints than that, or several roots, are refused (the model's grid).
- The skeleton is posed from a canonical stance (hips grounded, facing +Z, scaled to a diameter of 2) and the result is
  turned back into the rig's own frame. The training data's rigs stand in an ordinary rest pose (a T-pose or a
  standing quadruped); a rig in an unusual rest pose is untested here.
- The animation moves the skeleton; how well the mesh follows depends on the skin weights (UniRig's are good on a
  humanoid; fine details such as fingers and cloth will not bend well).
- The spectral position of a joint comes from an eigendecomposition whose basis is arbitrary up to sign; the engine
  fixes signs the way the reference does, but a rig with repeated eigenvalues (a perfectly symmetric one) can still
  get another basis, and that changes the motion a little.
- Rig frames are read back from the GLB: rotations are local to each joint's parent, written as a rotation track per
  joint and a translation track on the root.

## Checking it

`nextsycl model3d check <dump> --model unimate` compares against the official code (reference/unimate/ref.py, float32,
same noise): flan-T5 3e-7 to 7e-7 relative RMS, one denoiser pass 1.3e-6 (10 blocks, 2.2M values), the whole adaptive
flow 1.2e-6 with the same 128 evaluations, the decoded motion 3e-7, the rest pose and graph features identical.
