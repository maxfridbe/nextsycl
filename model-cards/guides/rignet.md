# Rigging with RigNet

RigNet gives a mesh a skeleton and skin weights: send a GLB or OBJ to `model3d.rig` (or `nextsycl model3d rig MESH
--model rignet`); the result is your mesh - its UVs, textures and vertices unchanged - with a glTF skin.

- **What it suits**: characters - humanoids, animals, cartoon figures - standing upright (+Y up), facing +Z, roughly
  symmetric about x = 0 (it mirrors the joints it finds on the left to the right). One connected, closed surface
  works best; loose parts and open shells rig too, less well.
- It rigs a copy decimated to about 4,000 triangles (`--opt-faces N`) and carries the weights back to every vertex of
  yours, four joints a vertex.
- **Too few or too many joints**: `--opt-threshold` (default 1e-5; lower keeps more joints) and `--opt-bandwidth`
  (default the learned 0.0429; the authors use 0.03 - 0.06 per model: smaller finds more, closer joints).
- `--seed` changes the surface samples behind the geodesic distances; results vary a little with it.
- Joints are named `root`, `joint_1`, `joint_2` ... in breadth-first order (a parent before its children).
- RigNet's code, and this engine, are GPLv3 (a commercial licence is offered by UMass).
