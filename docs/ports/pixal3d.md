## Pixal3D research report (only read the web, no files changed; the largest download was a 3 MB GGUF header)

**The repo URL had a typo.** The correct one is https://huggingface.co/Aero-Ex/Pixal3D-GGUF. Its README says only "Base Repo: https://github.com/Aero-Ex/ComfyUI-Trellis2-GGUF".

### What Pixal3D is
- "Pixal3D: Pixel-Aligned 3D Generation from Images" by Dong-Yang Li, Wang Zhao, Yuxin Chen, Wenbo Hu, Meng-Hao Guo, Fang-Lue Zhang, Ying Shan and Shi-Min Hu (Tsinghua, Tencent ARC, VUW). It was accepted to SIGGRAPH 2026. MIT licence.
  - Paper: arXiv 2605.10922
  - Code: https://github.com/TencentARC/Pixal3D
  - Weights: https://huggingface.co/TencentARC/Pixal3D
  - Project page: https://ldyang694.github.io/projects/pixal3d/
- **It is built on TRELLIS.2.** The `main` branch is a TRELLIS.2 fork: the package `pixal3d/` is TRELLIS.2's code with the same models, VAEs, samplers and normalisation stats. The `paper` branch is the Direct3D-S2 version. TencentARC/Pixal3D-D holds the Direct3D-S2 weights, which we can ignore.
- **What is new is "proj" conditioning.** Each block keeps TRELLIS.2's global cross-attention, but it only attends to the CLS token plus 4 register tokens. Then `out = cross_attn(x, global) + proj_linear(z_proj[voxel])`.
  - `z_proj` comes from projecting every grid or voxel centre into the image with a pinhole camera (FOV `camera_angle_x`, distance, mesh_scale, a Blender-style front view). The features are then sampled bilinearly with grid_sample from the DINOv3 patch map.
  - Shape and texture stages concatenate two samples, giving 2048 channels: the low-res DINOv3 patches and the same features upsampled by **NAF** (valeoai/NAF, which uses NATTEN neighbourhood attention).
  - The sparse-structure stage uses only the 1024-channel DINOv3 samples.
- **Camera.** By default, FOV comes from MoGe-2 (`Ruicheng/moge-2-vitl`) and distance is solved from it. A manual FOV is allowed.
- **Several views.** A separate set of `_mv` weights exists (`pipeline_mv.json`, `inference_mv.py`). It takes V views, each with its own camera-to-world matrix, projects all of them into the shared grid and **averages** `z_proj`, so the tensor shapes are unchanged. **These multi-view weights are not in Aero-Ex's GGUF repo**; it holds only the single-view weights.

### Every file in Aero-Ex/Pixal3D-GGUF
- **Root:** `pipeline.json` (a `Trellis2ImageTo3DPipeline` config) and `README.md`.
- **Sparse/** (`SparseStructureFlowModel`, 16³ dense):
  - `ss_flow_img_dit_1_3B_64_bf16.gguf`: 2.68 GB
  - `_Q8_0`: 1.43 GB
  - `_Q6_K`: 1.10 GB
  - `_Q5_K_M`: 0.93 GB
  - `_Q4_K_M`: 0.76 GB
- **shape/** (`ElasticSLatFlowModel`, input 32 channels): `slat_flow_img2shape_dit_1_3B_512_bf16` and `..._1024_bf16`, each at the same sizes:
  - BF16: 2.78 GB
  - Q8_0: 1.48 GB
  - Q6_K: 1.14 GB
  - Q5_K_M: 0.96 GB
  - Q4_K_M: 0.78 GB
- **texture/** (input 64 = noise plus shape latent, output 32): `slat_flow_imgshape2tex_dit_1_3B_1024_bf16`, at the same five sizes as shape/.
- **decoder/** (fp16 safetensors, not quantised):
  - `ss_dec_conv3d_16l8_fp16`: 0.148 GB
  - `shape_dec_next_dc_f16c32_fp16` (`FlexiDualGridVaeDecoder`): 0.948 GB
  - `tex_dec_next_dc_f16c32_fp16` (`SparseUnetVaeDecoder`, 6 PBR channels): 0.948 GB
- Every model has a `.json` config next to it.

**The "5 entries" are most likely the five quant levels** (BF16, Q8_0, Q6_K, Q5_K_M, Q4_K_M), which is exactly the node's `model_format` dropdown. One run actually needs the 4 flow models and the 3 decoders, plus these helpers that are not in the repo:
- DINOv3 ViT-L/16 (`facebook/dinov3-vitl16-pretrain-lvd1689m`)
- NAF (`naf_release.pth`)
- MoGe-2 ViT-L
- BiRefNet (RMBG-2.0)

**GGUF format.** The header says arch "flux" (the ComfyUI-GGUF convention), with PyTorch state_dict tensor names such as `blocks.N.cross_attn.cross_attn_block.to_kv` and `blocks.N.cross_attn.proj_linear.weight [2048→1536]`. In the Q4_K_M file there are 700 tensors: 244 Q4_K, 121 F16 (input layer, RMS-norm gammas) and 335 F32 (biases, modulation).

### Pipeline (`1536_cascade` by default, `1024_cascade` in low-VRAM mode)
1. Remove the background (BiRefNet). Run MoGe-2 for FOV and distance.
2. **Sparse structure.** A dense DiT on 16³ tokens: 30 blocks, width 1536, 12 heads, MLP 8192, RoPE, QK RMS-norm, shared adaLN. Then a conv3d decoder gives 64³ occupancy. Conditioning uses DINOv3 at 512 px.
3. **Shape at 512.** A sparse DiT, conditioned on DINOv3 512 px plus NAF upsampled to 512 on a 32³ grid.
4. The shape decoder's `upsample` gives high-res coordinates, which are re-quantised to res/16 under `max_num_tokens` (49152). The resolution steps down by 128 until the token count fits.
5. **Shape at 1024 or 1536.** Conditioned on DINOv3 at 1024 px plus NAF at 512.
6. **Texture at 1024.** Conditioned on DINOv3 at 1024 px plus NAF at 1024.
7. Decode with FlexiDualGrid to get the mesh, and with the sparse UNet to get PBR voxels. `o_voxel.postprocess.to_glb` then remeshes, decimates to 1M faces, UV-unwraps and bakes a 4096 texture.

**Default sampler settings** (FlowEuler with guidance interval, 12 steps each, sigma_min 1e-5):

| Stage | CFG | Rescale | Interval | rescale_t |
|---|---|---|---|---|
| Sparse structure | 7.5 | 0.7 | [0.6, 1] | 5 |
| Shape | 7.5 | 0.5 | [0.6, 1] | 3 |
| Texture | 1.0 | 0 | [0.6, 0.9] | 3 |

- Seed 42. The negative condition is all zeros for both the global and proj parts.

### What it shares with TRELLIS.2
Nearly everything carries over from a TRELLIS.2 port:
- The decoders are identical files. Pixal3D's README links them to TRELLIS.2.
- The DiT block layout, ElasticSLatFlow, RoPE, adaLN, sampler, normalisation stats, cascade and token-budget logic, and the GLB postprocess are all TRELLIS.2's.

What is new for us to build:
- A `proj_linear` per block (2048 or 1024 → 1536, plus an add after cross-attention).
- The projection grid and grid_sample step.
- Switching the image encoder from DINOv2 to DINOv3, with global context of only CLS plus 4 registers.
- **NAF**, which is a guided neighbourhood-attention upsampler. ComfyUI's fallback mode skips it, at some cost in quality.
- MoGe-2, or a manual FOV instead.

### The ComfyUI node pack and how it loads GGUF
- **Aero-Ex/ComfyUI-Trellis2-GGUF:** `Trellis2LoadModel_GGUF` has `modelname` set to "TRELLIS.2-4B" or "Pixal3D-GGUF" (the latter sets `isPixal3D`, the repo and the folder map in `model_manager.py`).
  - Other nodes: `...ImageCondGenerator`, `SparseGenerator`, `ShapeCascadeGenerator`, `TextureSlatGenerator`, `DecodeLatents`, `MeshWithVoxelMultiViewGenerator`.
  - The multi-view node takes front, back, left and right images with a `blend_temperature`. That is Aero-Ex's own blending, not the official `_mv` weights.
  - Loading goes through `gguf` plus ComfyUI-GGUF's ops/dequant, or `utils/gguf_utils.py` as a fallback; both dequantise on the fly. It runs MoGe and NAF through `torch.hub`.
- **Other ComfyUI routes:**
  - The official community node (bf16, with NATTEN/NAF modes): https://github.com/Saganaki22/Pixal3D-ComfyUI
  - ComfyUI's built-in route (`pixal3d_bf16` / `int8`, `dino_v3_L_naf`, `trellis_2_*_vae`): https://huggingface.co/Comfy-Org/Pixal3D
- **A port with no PyTorch already exists:** https://github.com/mousebar01/Pixal3D.cpp, with weights at https://huggingface.co/mousebar01/Pixal3D.cpp-GGUF (F16 packs for both the base and multi-view flows). It is a useful second reference for us.

### Hard ops
- Sparse submanifold 3D conv (`flex_gemm`/spconv) in the ConvNeXt-3D decoder blocks.
- Channel-to-space subdivision (`SparseResBlockC2S3d`).
- Variable-length full attention over up to about 49k tokens, with 3D RoPE.
- FlexiDualGrid mesh extraction.
- NATTEN 2D neighbourhood attention (in NAF).
- grid_sample.
- The postprocess uses cumesh/o_voxel for remesh, decimation and UV, plus nvdiffrast rasterisation for texture baking. These are all CUDA-only today.

### VRAM and runtime
- Each flow model is 1.3B parameters; together the four bf16 flows are about 22 GB.
- The official low-VRAM mode offloads stage by stage. Saganaki reports 12 GB cards hitting out-of-memory unless they use `hybrid_low_vram` or a smaller NAF size.
- I found no published runtime figures. Expect TRELLIS.2-class times, with about 1.3× the token cost at 1536.

### Python reference path for parity dumps
Use TencentARC/Pixal3D `inference.py --image X --fov <rad> --seed 42` with `ATTN_BACKEND=sdpa`, and pass `--fov` so MoGe is out of the loop.
- Hook points:
  - `DinoV3ProjFeatureExtractor.forward` for `z_global` and `z_proj`
  - each `sampler.sample` call
  - `shape_slat_decoder.upsample`
  - `decode_latent`
- For multi-view, use `inference_mv.py` with `pipeline_mv.json`.
- Aero-Ex's `trellis2_gguf/` package reproduces the quantised path.