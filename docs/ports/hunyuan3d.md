## Hunyuan3D porting brief (checked 2026-10-16 against HF trees, config files, tensor headers and GitHub source)

### (a) Hunyuan3D 2.1, https://huggingface.co/tencent/Hunyuan3D-2.1, code at https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1

**Shape files**
- `hunyuan3d-dit-v2-1/model.fp16.ckpt` is 7.37 GB, fp16, pickle-zip. It holds three parts:
  - DiT: 3.051B parameters
  - VAE with encoder: 0.328B
  - DINOv2-L conditioner: 0.304B
- Prefer **Comfy-Org/hunyuan3D_2.1_repackaged/hunyuan_3d_v2.1.safetensors** (7.37 GB). It has the same 1601 tensors and the same `model.`/`vae.`/`conditioner.` prefixes, but no pickle.
- `hunyuan3d-vae-v2-1/model.fp16.ckpt` (656 MB) is only needed for encoding a mesh. Image-to-3D doesn't use it.

**Conditioner.** This is DINOv2-**Large**, not giant: hidden 1024, 24 layers, 16 heads, patch 14, input 518 px. It produces 1370 tokens (37² plus CLS).
- Preprocessing (ImageProcessorV2): recenter the object on a 512 canvas with border_ratio 0.15, composite on white, resize to 518, then ImageNet mean/std.
- The unconditional input is all zeros.

**DiT (`HunYuanDiTPlain`)**
- Size: hidden 2048, depth 21, 16 heads (head dim 128). Input is 64 channels × **4096 latent tokens**.
- Attention: RMS qk-norm, no qkv bias, no positional embedding.
- Each block runs self-attention, then cross-attention to DINO tokens (K/V 1024→2048), then the MLP (×4).
- The timestep is not used for adaLN. It goes through an MLP and is **prepended as a token**.
- U-ViT long skips: blocks 11–20 concatenate a skip input (`skip_linear` 4096→2048, then `skip_norm`).
- **MoE in the last 6 blocks (15–20):** 8 experts, top-2, plus a shared expert. Experts are GELU FFNs 2048→8192, and the gate is 8×2048 with weights renormalized to sum to 1.

**Sampler**
- Defaults: FlowMatchEuler, 50 steps, **CFG 5.0** (batch doubled as cond then uncond), box 1.01, octree_resolution 384, num_chunks 8000, mc_level 0.
- Tencent's time runs in reverse: sigmas = linspace(0, 1, N), t/1000 is fed to the model, and the update is `x += (σ_next − σ)·v`.
- ComfyUI maps this as t → 1−t and negates the output. Use either convention as a cross-check.

**ShapeVAE decoder (3DShape2VecSet style)**
1. `post_kl` maps 64→1024.
2. 16 self-attention blocks run at width 1024 with 16 heads, qk-norm (LayerNorm on head dim 64), over the 4096 tokens.
3. `geo_decoder`: query points get a Fourier embedding (8 freqs, giving 51 dims) and a linear projection, then one cross-attention block from the queries to the latents (MLP ×4, `ln_post`), then a 1-logit output.
- Decoding: query a dense grid of (res+1)³ points in chunks. The optional Hierarchical/FlashVDM decoding uses coarse-to-fine levels down to res ≥ 63, refining only a dilated band near the surface.
- Meshing: marching cubes via skimage (`mc`), or `dmc` via **diso DiffDMC, which is CUDA-only**.
- Post-processing (pymeshlab): FloaterRemover, DegenerateFaceRemover, FaceReducer.

**Background removal.** `rembg.new_session()` uses the default **u2net** ONNX (~176 MB). It is not BiRefNet; BiRefNet is an optional swap.

**Paint model (`hunyuan3d-paintpbr-v2-1`, diffusers layout, mostly `.bin` pickles)**
- Files:
  - `unet/diffusion_pytorch_model.bin`: 3.93 GB. It holds the main UNet plus `unet_dual`, a reference UNet (SD2.1 config: 320/640/1280/1280, cross-attention dim 1024, linear projection).
  - `vae/…bin`: 335 MB (SD AutoencoderKL).
  - `text_encoder` (1.36 GB) and `image_encoder` (CLIP ViT-H, 1.26 GB) are loaded by `model_index` but **unused at inference**. Learned per-material text tokens replace the prompt.
- `UNet2p5DConditionModel` changes to the SD2.1 UNet:
  - `conv_in` takes **12 channels**: noisy latent plus VAE latents of the normal map and the position map.
  - Two material streams, albedo and metallic-roughness (`mr`), coupled by material-aware attention.
  - Multiview attention across views, with position RoPE.
  - Reference attention to the `unet_dual` features of the input image.
  - DINOv2 features from a separate download, **facebook/dinov2-giant** (`model.safetensors`, 4.55 GB fp32).
- Scheduler: v-prediction with zero-terminal-SNR betas, run as **UniPC with 15 steps**, trailing spacing.
- Guidance is 3.0 with **3-way CFG**: (no ref + zero DINO, ref only, full). It is combined as `u + g·s·(ref−u) + g·s·(full−ref)`, with a per-view scale s.
- Views: 6 at 512 px by default (`max_num_view=6, resolution=512`), chosen from 6 fixed cameras plus 24 at ±20° elevation.
- The VAE encode uses `posterior.sample()`, which is random. Parity code must use the mode or dumped latents.

**Texture pipeline steps**
1. Optional remesh (pymeshlab).
2. UV unwrap with **xatlas**.
3. Render normal and position maps.
4. Multiview diffusion.
5. **RealESRGAN_x4plus** upscale (67 MB, from the GitHub release).
6. Bake each view into a 4096² UV texture from 2048² renders (`bake_exp` 4).
7. Inpaint missing texels: `mesh_inpaint_processor.cpp` (CPU vertex propagation), then `cv2.inpaint` (Navier-Stokes).
8. Export OBJ/GLB.

**Custom ops and portable replacements**

| Op | What it is | Portable replacement |
|---|---|---|
| `custom_rasterizer` (`rasterizer_gpu.cu`) | z-buffer of `int64 (quantized_z·MAXINT + face_id+1)` with `atomicMin`, then a barycentric/face-index pass | One SYCL kernel with a 64-bit atomic fetch_min, or the existing CPU code path (`rasterizer.cpp`, dispatched when tensors are on CPU) |
| diso DMC | CUDA dual marching cubes | Plain marching cubes (CPU or SYCL) |
| xatlas | C++ UV unwrap | xatlas C API through Rust FFI |
| `mesh_inpaint_processor`, cv2 NS inpaint, pymeshlab | CPU code | Port to Rust |
| flash-attn | Not required | The code uses PyTorch SDPA |

**VRAM and download size**
- README figures: 10 GB for shape, 21 GB for texture, 29 GB for both (CUDA, fp16). On a 32 GB B70, load the stages sequentially.
- Total download is about 7.4 GB (shape) + 4.3 GB (paint UNet/VAE, about 6.9 GB for the whole paint folder) + 4.55 GB (dinov2-giant) + 0.25 GB (ESRGAN, u2net), roughly **17–19 GB**.

### (b) "2mini-turbo"
- **Exact source:** **tencent/Hunyuan3D-2mini, subfolder `hunyuan3d-dit-v2-mini-turbo`**, file `model.fp16.safetensors` (3.82 GB; a `.ckpt` copy also exists).
  - Contents: DiT 0.561B, **DINOv2-giant 1.136B**, VAE 0.214B.
  - Add **`hunyuan3d-vae-v2-mini-turbo/model.fp16.safetensors`** (407 MB), the FlashVDM decoder: `geo_decoder` width 512, expand 1, downsample 2, no `ln_post`.
  - **Total about 4.23 GB.** Both are shape-only; there is no texture.
- **2mv is a different model:** https://huggingface.co/tencent/Hunyuan3D-2mv is the multiview-input (front/left/back/right) **1.1B** DiT with 16 double and 32 single blocks. Its turbo variant is `hunyuan3d-dit-v2-mv-turbo`, 4.93 GB.
- **Architecture vs 2.1:** a Flux-style MMDiT (`Hunyuan3DDiT`).
  - Hidden 1024, 16 heads (head 64), **8 double-stream + 16 single-stream blocks**, qkv bias, adaLN modulation, no RoPE (pe = None). There are no skips and no MoE.
  - `cond_in` maps 1536→1024. **DINOv2-giant** (40 layers, SwiGLU) runs at **1022 px** with interpolated position embeddings: 73² + 1 = 5330 cond tokens. The mini and mini-fast variants use 518 px.
  - `guidance_in` (distilled guidance): there is no CFG doubling, and guidance_scale 5.0 goes into the embedding.
  - The timestep embedding uses max_period = 1000 with t in [0,1]. ComfyUI notes this as an upstream bug that must be reproduced.
- **Latent token count, two conflicting values:**
  - The turbo DiT config says 3072 tokens, but `enable_flashvdm()` replaces the VAE with the mini-turbo VAE, and `prepare_latents` reads `vae.latent_shape`, which is **512**.
  - So the official example appears to sample 512 tokens. Confirm this with a reference dump before relying on it.
- **Sampler:** ConsistencyFlowMatchEuler (pcm_timesteps 100), **5 steps**, octree_resolution 380, num_chunks 20000.
  - Official example: https://github.com/Tencent-Hunyuan/Hunyuan3D-2/blob/main/examples/faster_shape_gen_with_flashvdm_mini_turbo.py with `enable_flashvdm(topk_mode='merge')`.
- **FlashVDM** (https://github.com/Tencent/FlashVDM, arXiv 2503.16302):
  - Hierarchical decoding: a coarse grid, then refinement of only a dilated near-surface band at each level up to 380.
  - Adaptive KV selection: the grid is split into 4³ mini-grids, and each mini-grid cross-attends only to the top-k latent tokens.
  - A lightweight decoder.
  - The paper claims the VAE decode is about 45× faster and the whole run takes about 1 s on a high-end NVIDIA GPU.
- **VRAM:** under 6 GB, an estimate based on the README's 6 GB figure for the 1.1B model.

### GGUF
- There is nothing for 2.1 or mini. city96, Aero-Ex and QuantStack have nothing.
- **calcuis/hy3d-gguf** only quantizes the **2.0 1.1B** DiT (q2–q8 and f16, the "pig" VAE). It isn't worth using: fp16 safetensors are small enough.

### ComfyUI native reference (compact, supports both)
- `comfy/ldm/hunyuan3d/model.py` (148 lines): `Hunyuan3Dv2`, built from Flux `DoubleStreamBlock`/`SingleStreamBlock`. It covers 2.0, mv and mini (depth 8); see `supported_models.py` `Hunyuan3Dv2mini`.
- `comfy/ldm/hunyuan3dv2_1/hunyuandit.py` (674 lines): `HunYuanDiTPlain` with MoE. It swaps the CFG halves because Tencent orders the batch cond-first.
- `comfy/ldm/hunyuan3d/vae.py` (988 lines): ShapeVAE plus `VanillaVolumeDecoder`, dense grid only, with no FlashVDM.
- Latent scale factors: 2.0 0.99909, **2.1 1.00395**, **mini 1.01881**. Shift is 1.0 and ModelType is FLOW.
- `comfy_extras/nodes_hunyuan3d.py`: `EmptyLatentHunyuan3Dv2` (default 3072 tokens), conditioning from a CLIP-Vision loader that auto-detects DINOv2 L/giant (`last_hidden_state`, zeros for negative), `VAEDecodeHunyuan3D` (chunks 8000, res 256), and `VoxelToMesh` (marching cubes or surface-net).
- **There is no paint/texture support in ComfyUI core.** For texturing see kijai's ComfyUI-Hunyuan3DWrapper (https://github.com/kijai/ComfyUI-Hunyuan3DWrapper).

### Python CPU reference for parity dumps
- **Shape:**
  - Clone the 2.1 repo and install CPU torch with the `hy3dshape` requirements.
  - Load with `Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2.1', subfolder='hunyuan3d-dit-v2-1', device='cpu')`, or `from_single_file(..., dtype=torch.float32)`. The code uses SDPA only.
  - Dump the initial noise (don't try to reproduce torch RNG), the conditioner output, per-block outputs via forward hooks, and the `geo_decoder` logits. Use few steps and octree_resolution 128 for speed.
  - For mini-turbo, use the 2.0 repo's `hy3dgen` with the same approach.
  - ComfyUI run with `--cpu` is an independent second reference.
- **Paint:**
  - Run the diffusers pipeline on CPU in fp32. It is slow: 2 streams × 6 views × 3 CFG branches × 15 steps.
  - `custom_rasterizer` has a CPU path, but `setup.py` builds a CUDAExtension. Build it as a CppExtension without the `.cu` file.

### URLs
- https://huggingface.co/tencent/Hunyuan3D-2.1
- https://huggingface.co/tencent/Hunyuan3D-2mini
- https://huggingface.co/tencent/Hunyuan3D-2mv
- https://huggingface.co/Comfy-Org/hunyuan3D_2.1_repackaged
- https://huggingface.co/facebook/dinov2-giant
- https://github.com/Tencent-Hunyuan/Hunyuan3D-2
- https://github.com/comfyanonymous/ComfyUI/tree/master/comfy/ldm/hunyuan3d
- https://github.com/comfyanonymous/ComfyUI/tree/master/comfy/ldm/hunyuan3dv2_1