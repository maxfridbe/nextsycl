## Porting brief: UniMate (Friedrich-M, SIGGRAPH Asia 2026). Code MIT, **checkpoints CC BY-NC 4.0** (say so in the card/guide)

Animation for a rigged asset: `model3d rig RIGGED.glb --model unimate --prompt "An object walks forward."` returns the asset with its
skeleton animated (glTF skeletal animation, not morph targets). Natural chain: `skintokens` (or `unirig`) rigs, UniMate animates.
Reference code (read only, never executed on hardac1): `https://github.com/Friedrich-M/UniMate` @ b78c780 (clone at the session scratchpad `unimate/repo`).

**Weights (`https://huggingface.co/Linzhan/UniMate`, rev 174fc755ec248d0c98f27759b7a95648e7f05905), one run directory each: `config.json`,
`dataset_stats.npy`, `checkpoints/checkpoint_step_N.pt`** (model + EMA + optimizer state; inference reads the EMA weights):
- default: `unimate_uniml3d_f60_v3` step 150000 (1,185,836,040 B), the `graph_adaln` variant; trained on UniML3D (7,430 skeletons)
- also `unimate_uniml3d_f60_v3_cross_attn` step 150000 (1,059,089,958 B), the `full_cross_attn` variant - a second card if cheap.
- Mixamo/Truebones-only runs exist; not needed.
- Text encoder `google/flan-t5-base` (T5 encoder, ~1 GB) from its own repo; pin a revision, check sha256.
The checkpoints are torch pickles: read with the safe readers, extract only the EMA tensors.

**Pipeline (`unimate/inference/sample.py`, `unimate/models/denoiser/**` incl. `rope.py`, `blocks/common.py`, `unimate/models/flow/transport.py`,
`unimate/utils/motion_utils.py`):** read the rig (joints, parents, rest pose, offsets, joint names), the per-joint features and names
(T5-pooled joint-name embeddings), the caption (T5, tokens for `cross_attn`, mean-pooled for `adaln`), then flow matching (velocity
network, ODE sampling, classifier-free guidance `cfg_scale` 3) over a (60 frames x joints x features) tensor, de-normalise with
`dataset_stats.npy`, convert to joint rotations / root motion, write the animation into the glTF.
**The hard part is the rig preprocessing** (`data_process/rig_preprocess/**`): it labels joints and facing with an LLM and a human review.
The engine cannot call an LLM: use the rule-based pieces (`joint_annotation/names_clean_rule.py`, `structure_rule.py`, facing from geometry),
and say plainly in the guide what happens for generic names such as `bone_0` (SkinTokens'/UniRig's output): the model may be run with the
structure only (graph distances / edge types / depth) - measure what that does to the motion before claiming it works.

**Where:** `model3d/unimate` (own crate, do not edit animateanymesh), arch `unimate`, `does: [animate]`; the contract may need a
skeleton-animation output (look at `model3d/contract`, the `rig` RPC and the GLB writer in `crates/mesh`). Kernels only in a silo
(`kernels/model3d/unimate/silo`). Card `model-cards/model3d/unimate.yml`, guide `model-cards/guides/unimate.md`, docs in `docs/3d.md`.
