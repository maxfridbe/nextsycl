## Porting briefs: UniRig, RigNet, AnimateAnyMesh

I checked file sizes against the HF tree API and with HTTP range reads of the checkpoint zip directories, so nothing was fully downloaded. The code repos are cloned under `/tmp/claude-1000/-var-home-maxfridbe-Dev-hardac/ca6b4189-dc54-4ec0-aa62-40001a0681e3/scratchpad/src/{UniRig,RigNet,AnimateAnyMesh}`. No project files were changed.

---

### 1. UniRig (VAST-AI, SIGGRAPH'25). License: MIT (HF card and repo)

**The 2 entries (`https://huggingface.co/VAST-AI/UniRig/resolve/main/<path>`):**
- `skeleton/articulation-xl_quantization_256/model.ckpt`: **1,439,617,174 B**. This is a Lightning ckpt holding only `state_dict`. It is fp32, so about 360M params.
- `skin/articulation-xl/model.ckpt`: **4,375,464,854 B**. It also contains `optimizer_states` and `lr_schedulers`, and the keys carry `_fsdp_wrapped_module.` and `_checkpoint_wrapped_module` prefixes. The real weights are probably about 1/3 of the file (an estimate). Strip those prefixes when converting.
- Also in the repo: `skeleton/rignet/model.ckpt` (621.7 MB, a variant trained on RigNet data) and datasets (`data/rigxl/processed.7z` 4.87 GB, `data/rignet/processed.zip` 164 MB). These are not needed.
- **Total for inference: about 5.82 GB.** You could re-save the skin model to about 1.5 GB.

**Pipeline** (`launch/inference/generate_skeleton.sh`, `generate_skin.sh`, `merge.sh`):
1. **Extract** (`src/data/extract.py`): imports obj/fbx/glb/vrm with **bpy 4.2 (Blender as a module)**, decimates to 50k faces with `fast_simplification`, and writes `raw_data.npz` (verts, vertex/face normals, faces). Coordinates are then normalized into [-1,1].
2. **Point sampling, "mix" method** (`src/data/sampler.py`):
   - Skeleton stage: 8192 vertex samples plus area-weighted barycentric surface samples, 65,536 points in total.
   - Skin stage: 32,768 points.
   - Each point carries a normal: the vertex normal, or the face normal for surface samples.
3. **Skeleton model** (`src/model/unirig_ar.py`):
   - **Encoder:** Michelangelo / 3DShape2VecSet `ShapeAsLatentPerceiverEncoder`.
     - Randomly takes 4×1024 points, then runs **FPS (torch_cluster.fps, ratio 1/4)** to get 1024 query tokens.
     - Fourier position embedding (8 freqs, no π) plus normals.
     - One cross-attention, then **16 self-attention layers**, width 512, 8 heads, ln_post.
     - `output_proj` 512→1024 gives 1024 prefix embeddings.
   - **Decoder:** HF **OPT-350m** config built from scratch (`facebook/opt-350m`): hidden 1024, pre-norm, n_positions 3076, vocab **267**:
     - 256 quantized coordinate bins over [-1,1]
     - branch / bos / eos / pad
     - spring
     - part tokens body and hand
     - cls_none
     - cls vroid / mixamo / articulationxl
   - **Sequence:** prefix, then `bos`, then `cls=articulationxl`, then the generated tokens.
   - A **grammar mask** (`VocabSwitchingLogitsProcessor` calling `tokenizer.next_posible_token`) must be ported exactly. It is CPU logic per step.
   - **Generation params** (`configs/system/ar_inference_articulationxl.yaml`): max_new_tokens 2048, **num_beams 15 with do_sample** (HF beam-sample mode), top_k 5, top_p 0.95, repetition_penalty 3.0, temperature 1.5.
   - **Output:** a "skeleton tree tokenization" that detokenizes into joints (J×3), parents and tails (`src/tokenizer/tokenizer_part.py`).
4. **Skin model** (`src/model/unirig_skin.py`, `configs/model/unirig_skin.yaml`):
   - **Per-point encoder:** **Point Transformer V3** (`ptv3obj`, 9-ch input: xyz, normal, zeros). Details:
     - grid 0.005
     - enc depths [3,3,3,6,16], channels [32..384], heads [2..24]
     - serialized orders z, z-trans, hilbert, hilbert-trans
     - patch size 1024
     - **spconv SubMConv3d** for positional encoding
     - serialized pooling via torch_scatter
     - qk-norm
     - must run in fp32 (the code notes sparse-conv precision bugs)
   - **Global encoder:** a second Michelangelo encoder (same configuration).
   - **BoneEncoder:** Fourier embedding of joint positions, an MLP to 768, then 8 residual cross-attention blocks (flash_attn `MHA`, 16 heads) over bones plus global latents.
   - **Mesh-bone attention:** 16 residual cross-attention blocks from mesh features to bone and global latents.
   - **Skin head:** a per-head q·k score (feat 768 × 16 heads) gives N×J×16. It is concatenated with the embedded **voxel_skin** prior, passed through an MLP to 1, then softmax over J.
     - The result is multiplied by voxel-mask^3 and renormalized.
     - Vertices are processed in chunks of 512.
   - **voxel_skin prior** (`src/data/vertex_group.py`): a 196³ voxelization built from **pyrender (OpenGL) or open3d depth renders**, followed by a grid geodesic-ish propagation (alpha 0.5, grid_query 7).
     - This is heavy CPU/GL preprocessing and must be reimplemented, for example as GPU ray casting.
5. **Output and writing:**
   - `predict_skeleton.npz` and `predict_skin.npz` are written.
   - **FBX/GLB/VRM export uses bpy** (`src/data/exporter.py` uses `bpy.ops.export_scene.fbx`; `src/inference/merge.py` exports fbx, gltf or vrm).
   - A Rust port should write glTF directly (skin, joints, inverseBindMatrices, JOINTS_0/WEIGHTS_0 keeping the top 4 weights).

**Hard ops:**
- FPS
- spconv submanifold conv plus z-order/Hilbert serialization (PTv3)
- flash_attn varlen in PTv3 patches and MHA (plain SDPA is fine)
- torch_scatter segment ops
- HF beam-sample with a grammar mask and repetition penalty
- the voxel-skin depth renderer

**VRAM/runtime:** the README says generation needs at least 8 GB VRAM. The skeleton stage takes seconds to tens of seconds (up to 2048 tokens × 15 beams); the skin stage is seconds.

**Parity:** the reference is the repo's Python (torch ≥2.3, transformers 4.51.3, spconv, flash_attn, bpy 4.2). Sampling is stochastic (random point sampling plus beam-sample), so:
- check the skeleton stage with teacher-forced logits on fixed points and fixed FPS indices;
- check the skin stage on a fixed skeleton and a fixed `voxel_skin`.

Note: the README points to the successor **SkinTokens** (VAST-AI-Research/SkinTokens).

---

### 2. RigNet (Xu et al., SIGGRAPH 2020). License: **GPLv3** (code; a commercial licence is offered by UMass)

**Weights:** not on HF. They are on Google Drive as `trained_models.zip`, **195,677,735 B (about 196 MB)**:
- `https://drive.google.com/file/d/1gM2Lerk7a2R0g9DwlK3IvCfp8c2aFVXs` (direct: `https://drive.usercontent.google.com/download?id=1gM2Lerk7a2R0g9DwlK3IvCfp8c2aFVXs&export=download&confirm=t`)
- It unpacks to `checkpoints/{gcn_meanshift,rootnet,bonenet,skinnet}/model_best.pth.tar`. Each is a torch tarball with a `state_dict`, from torch 1.12.
- The dataset is a separate Drive file; skip it.

**Preprocessing** (`quick_start.py create_single_data`):
- Remesh to **1K–5K vertices** first. The authors use MeshLab quadric edge collapse; the input must be named `*_remesh.obj`.
- Normalize, then compute vertex normals (open3d 0.9).
- Node features: xyz plus normal (6).
- **Topology edges** come from the faces, with self-loops.
- **Surface geodesic** (`geometric_proc/common_ops.calc_surface_geodesic`):
  - Poisson-disk sample 4000 points.
  - Build a 5-NN graph filtered by normal cosine > -0.5.
  - Run **all-pairs Dijkstra** (scipy). Inf distances are replaced by Euclidean + 8.
- **Geodesic edges:** neighbours within a geodesic ball.
- **Voxelization:** shells out to the bundled **`binvox` binary, 88³**, used for the inside check of bones.

**Networks** (all from the `GCU` block in `models/gcn_basic_modules.py`):
- **GCU:** two EdgeConvs, one over topology edges and one over geodesic edges. Each EdgeConv is MLP([x_i, x_j−x_i]) with **max aggregation**. Every MLP layer is Linear→ReLU→BatchNorm.
- **JointNet** (`GCN.py`, `JOINTNET_MASKNET_MEANSHIFT`):
  - 3 GCUs (64/256/512), a global max-pool to 1024, then an MLP.
  - Outputs per-vertex displacement, attention and a learned bandwidth (0.0429).
  - Then attention-weighted **mean-shift** (40 iterations), a density threshold (default 1e-5, tuned per model) and NMS give the joints.
- **RootNet** (`ROOT_GCN.py`): a GCU shape encoder plus a PointNet++ joint encoder (SAModule: **fps, radius, PointConv**; GlobalSA; FP with **knn_interpolate**). It classifies the root joint.
- **BoneNet** (`PairCls_GCN.py`): pairwise connection probabilities. The cost −log p, plus a penalty for bones outside the voxels, feeds **symmetric Prim MST** (CPU), giving the hierarchy.
- **SkinNet** (`SKINNING.py`, nearest_bone=5, Dg and Lf flags):
  - **Volumetric geodesic** from each vertex to each bone: trimesh **ray-mesh visibility** tests plus the surface geodesic for occluded points. This is the slowest step.
  - Per-vertex input: 5 nearest bones × (bone endpoints, 1/dist, is-leaf).
  - MLP, then 3 GCUs (512/256/256), then an MLP; softmax over the 5 bones.
  - Post-processing: one-ring smoothing, zero out weights below 0.35×max, renormalize.

**Output:** `*_rig.txt` containing joint, root, hier and skin lines. FBX comes only through `maya_save_fbx.py` (mayapy). For the port, write glTF directly.

**Hard ops:**
- EdgeConv message passing (gather, MLP, scatter_max)
- global max-pool
- FPS, radius search, knn_interpolate
- BatchNorm in eval form (fold it into the Linear layers)

The networks are tiny. The hard part is the geometry: 4000² Dijkstra, ray casting, binvox and mean-shift.

**Runtime:** network inference is milliseconds. Geodesic plus visibility take tens of seconds to minutes on CPU (the "takes some time" step). VRAM is under 1 GB.

**Parity:** the repo's Python stack is torch 1.12, torch-geometric 1.7.2 and open3d 0.9. Poisson sampling is random, so fix the samples or dump intermediates.

---

### 3. AnimateAnyMesh (Wu et al., ICCV'25, HUST/DAMO). License: **Apache-2.0** (HF and GitHub `JarrentWu1031/AnimateAnyMesh`)

**What it does:** takes a static GLB mesh and a text prompt and produces **16 frames of per-vertex trajectories**. It is feed-forward and runs in seconds. The successor is AnimateAnyMesh++ (arXiv 2604.26917).

**Files (`https://huggingface.co/JarrentWu/AnimateAnyMesh/resolve/main/<path>`):**
- `rf_model/rf_epoch_f.pth`: **805,255,167 B**, fp32. It **includes the full SD-1.5 CLIP text encoder** (346 tensors, about 123M params, about 492 MB), so the DiT itself is about 78M params.
- `dvae/dvae_f.pth`: **146,248,555 B**, fp32, about 36.5M params.
- `rf_model/training_config.json`, `dvae/model_config.json`, `dvae_factors/dvae_f.json`. The latent rescale stats are f0 mean 0.2052 / std 9.0644 and ft mean 0.9568 / std 6.3974.
- The tokenizer comes from `stable-diffusion-v1-5/stable-diffusion-v1-5/tokenizer` (vocab.json and merges.txt, about 1.5 MB). The code also loads the text_encoder from there, but the checkpoint already contains it.
- **Total: about 952 MB.** A Google Drive mirror exists (`1_ixt6pWlUpFvwFn6eV3xuOf1G7g6ijfo`).

**DyMeshVAE** (`networks/dymeshvae.py`): dim 256, enc and dec depth 8, latent 32+32, T=16, `output_dim` 48 (= 16×3), n_layers 1.
- **Encoding:**
  - PointEmbed (Fourier basis, then 256) of rest vertices.
  - **NeighborAttention** with a dense **N×N adjacency mask** from the faces (1 head, dim 512).
  - **pytorch3d `sample_farthest_points` on the 256-d embeddings** chooses `num_traj` vertices (default max(512, N/8)).
  - 8 "sync" cross-attention blocks: q from the sampled points, k from all vertices, two value streams (rest shape and trajectory) concatenated in one SDPA, then GEGLU FFNs.
  - x0 = mean_fc (deterministic). xt comes from a Gaussian sample but is replaced by RF noise at inference.
- **Decoding:** 8 sync self-attention blocks (8 heads × 64) over the latent tokens. Then every vertex queries them (cross-attention, dim 512, 1 head) and the result goes through `to_outputs`, giving a 16×3 offset added to the rest position.

**RF DiT** (`networks/transformer.py DyMeshMMDiT`): width 512, 12 layers, 8 heads, in/out 64 channels.
- CogVideoX-style **joint attention** over [trajectory tokens + 77 CLIP tokens], with separate adaLN-Zero modulations (from t-emb) for each stream.
- CLIP last hidden state (768) goes through a Linear to 512.
- `use_flash2` means the flash_attn package.

**Sampler** (`diffusion/rf_diffusion.py`):
- torchdiffeq `odeint` with **midpoint**, on a fixed grid of `linspace(0,1,64)`. That is 63 steps × 2 evaluations = **126 DiT forwards at batch 2** (CFG; uncond = empty prompt with zeroed text embedding).
- guidance 3.0 (the README suggests up to 5).
- The f0 channels are clamped to the encoded rest-shape latents at every evaluation. Only the motion channels are generated.

**Inputs and outputs:**
- Input is GLB only, imported with **bpy**. Identical vertices are merged; the mesh is recentred and scaled to [-1,1] and padded to max(4096, N).
- Output is a [16, N, 3] tensor. **Export is Blender shape keys, one per frame (Frame_i) with keyframed values, to FBX or ABC (Alembic)**, plus a rendered video.
- No GLB is written. glTF morph targets plus a weights animation map naturally onto this.

**Hard ops:**
- FPS over 256-d features (pytorch3d)
- the **dense N×N masked attention** for adjacency. At 20k vertices that is a 400M-entry mask. Rewrite it as sparse neighbour attention over a CSR of face edges.
- SDPA with the concatenated-V trick
- the CLIP-L text encoder
- everything in fp32

**VRAM/runtime:** "a few seconds" per the paper. Activations are small (512 to about 2.5k tokens). The dense adjacency mask dominates memory for large meshes.

**Parity:** reference is `test_drive.py` (torch 2.6, transformers 4.55.3, pytorch3d, torchdiffeq 0.2.5). Seed the noise and dump the FPS indices so the Rust run uses the same ones.

---

**Totals:**

| Model | Download | License |
|---|---|---|
| UniRig | about 5.82 GB (2 ckpts) | MIT |
| RigNet | about 196 MB (Drive zip) | GPLv3 |
| AnimateAnyMesh | about 952 MB | Apache-2.0 |

**Shared porting need:** none of the three writes FBX or GLB without Blender or Maya, so the Rust side needs its own glTF writer (skins for UniRig and RigNet, morph targets for AnimateAnyMesh).