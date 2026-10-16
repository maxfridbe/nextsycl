# TRELLIS.2 porting brief for nextsycl

Sources checked: `microsoft/TRELLIS.2` @75fbf01, `pwilkin/trellis.cpp` @610c90e (2026-10-11), the `Aero-Ex/Trellis2-GGUF` tree and configs, and the `Aero-Ex/ComfyUI-Trellis2-GGUF` loader. I read GGUF and safetensors headers with HTTP range requests only; no large downloads. Clones are in `/tmp/claude-1000/-var-home-maxfridbe-Dev-hardac/ca6b4189-dc54-4ec0-aa62-40001a0681e3/scratchpad/{t2,tcpp,aero}`.

## 1. Files

Base URL: `https://huggingface.co/Aero-Ex/Trellis2-GGUF/resolve/main/<path>`

| path | size | needed? |
|---|---|---|
| `pipeline.json` | small | yes: sampler params and the 32-channel mean/std vectors |
| `texturing_pipeline.json` | small | only for texture-only mode |
| `Vision/dinov3-vitl16-pretrain-lvd1689m.safetensors` (F32, 415 tensors) + `Vision/preprocessor_config.json` | 1213 MB | yes: image encoder |
| `decoders/Stage1/ss_dec_conv3d_16l8_fp16.{json,safetensors}` (F16) | 148 MB | yes |
| `decoders/Stage2/shape_dec_next_dc_f16c32_fp16.{json,safetensors}` (F16) | 949 MB | yes |
| `decoders/Stage2/tex_dec_next_dc_f16c32_fp16.{json,safetensors}` (F16) | 949 MB | yes |
| `refiner/ss_flow_img_dit_1_3B_64_bf16{.safetensors,_Q4_K_M,_Q5_K_M,_Q6_K,_Q8_0.gguf,_fp8.safetensors}` + `.json` | 2584 / 789 / 948 / 1117 / 1426 / 1292 MB | yes, pick one |
| `shape/slat_flow_img2shape_dit_1_3B_{512,1024}_bf16*` (same 6 variants each) | same sizes | 512 = light path and cascade first pass; 1024 = cascade second pass |
| `texture/slat_flow_imgshape2tex_dit_1_3B_{512,1024}_bf16*` | same | 512 for the res-512 path, 1024 for the cascade |
| `encoders/{shape,tex}_enc_next_dc_f16c32_fp16.*` | 709 MB each | no: only for texturing an existing mesh |
| `test/*.gguf` (BF16 experiments), `test/trellis_ss_bf16.json` | ~2.6 GB each | no |

**The Aero-Ex GGUFs are in ComfyUI-GGUF format.**
- The header says `general.architecture=flux`; `file_type 15` means Q4_K_M. There is no config inside the file; the config is in the sibling `.json`.
- Linear weights are Q4_K/Q5_K/Q6_K/Q8_0, norm gammas are BF16, biases are F32.
- Key names are the original ones except `adaLN_modulation.1.*`, which is stored as `guidance_in.*`. The loader's remap table is at `aero/trellis2_gguf/utils/gguf_utils.py:665`.
- The fp8 safetensors are F8_E4M3 with a per-tensor `*_scale`.

**BiRefNet is not in the Aero-Ex repo.** Their ComfyUI node uses `rembg` instead.

**The "10 manifest entries" are the trellis.cpp download set from `ilintar/trellis2-gguf`.** URLs are `https://huggingface.co/ilintar/trellis2-gguf/resolve/main/[q4/|q8/]<name>`:
- `birefnet.gguf`: 883 MB, F32, BiRefNet Swin-L, from `ZhengPeng7/BiRefNet`
- `dinov3.gguf`: 607 MB
- `ss_flow`: 2587 MB
- `ss_dec`: 147 MB
- `shape_flow_512`, `shape_flow_1024`: 2587 MB each
- `shape_dec`: 949 MB
- `tex_flow_512`, `tex_flow_1024`: 2587 MB each
- `tex_dec`: 949 MB

The full set is about 16.5 GB; q8 is about 9.5 GB, q4 about 6 GB. These files use their own format: architecture `trellis2-*`, the config embedded as `trellis.config_json`, quants Q4_0/Q8_0.

Upstream sources:
- `microsoft/TRELLIS.2-4B` (all ckpts)
- `microsoft/TRELLIS-image-large` (`ss_dec`)
- `facebook/dinov3-vitl16-pretrain-lvd1689m` (gated; ungated mirrors are `timm/vit_large_patch16_dinov3.lvd1689m` and `Aero-Ex/Dinov3`)
- `briaai/RMBG-2.0` (gated) or `ZhengPeng7/BiRefNet`

## 2. Pipeline

The entry point is `trellis2/pipelines/trellis2_image_to_3d.py`. Every flow model shares one DiT: 30 blocks, width 1536, 12 heads × 128, MLP 8192 with GELU-tanh, about 1.3B parameters.
- Modulation (AdaLN): one shared `adaLN_modulation.1` (1536→9216) plus a per-block `modulation[9216]`, split 6 ways.
- Each block: self-attention, then cross-attention to the image tokens, then the MLP. Q and K get RMSNorm in both attentions.
- Self-attention uses a 3D RoPE that rotates adjacent pairs, not ggml's half-split RoPE. Frequencies are 21 per axis for x, y, z plus one identity pair.
- Attention is always full over all tokens. The sparse flows use `attn_mode='full'`, so there is no windowing in inference.

0. **Preprocess.**
   - Downscale so the long side is at most 1024 (Lanczos).
   - If the image has no alpha, run BiRefNet.
   - Take the bounding box of alpha > 0.8 and crop it to a centered square.
   - Premultiply RGB by alpha (black background).
1. **Image encoder: DINOv3 ViT-L/16.**
   - 24 layers, 16 heads × 64, RoPE on patch tokens.
   - The image is resized to the working resolution, not 224: 512 gives 1029 tokens (CLS + 4 registers + 1024 patches); 1024 gives 4101.
   - Final step is a parameter-free `F.layer_norm`.
   - The negative (unconditional) input for guidance is all zeros.
2. **Sparse-structure flow.** Dense 16³ grid = 4096 tokens, 8 channels in and out, conditioned on the 512 image tokens.
3. **Sparse-structure decoder (dense conv3d).**
   - 8→512 at 16³, then 2 middle ResBlocks, then upsample to 32³ at 128 channels and 64³ at 32 channels, then 1 output channel.
   - Uses `pixel_shuffle_3d` and a per-voxel channel LayerNorm.
   - Logits > 0 give occupancy at 64³. For the 512 and cascade paths, max-pool down to 32³. `argwhere` gives the coordinates.
4. **Shape latent flow (sparse).** 32 channels per active voxel.
   - Cascade first pass: `flow_512` with the 512 image tokens.
   - `shape_dec.upsample(x4)` then predicts child voxels.
   - The coordinates are quantized with `((c+0.5)/512*(res//16))` and made unique.
   - If there are 49152 or more tokens, the target resolution drops by 128 and the step repeats, with a floor of 1024.
   - Second pass: `flow_1024` with the 1024 image tokens.
   - Then denormalize with the shape mean/std from `pipeline.json`.
5. **Texture latent flow.**
   - Input is 64 channels: 32 noise channels followed by the 32-channel shape latent, renormalized. Output is 32 channels.
   - Guidance strength is 1, so each step is a single conditioned pass.
   - Then denormalize with the texture mean/std.
6. **Decoders.**
   - **Shape:** `FlexiDualGridVaeDecoder`.
     - `from_latent` 32→1024.
     - Stage channels [1024, 512, 256, 128, 64], with [4, 16, 8, 4, 0] SparseConvNeXt blocks per stage.
     - Each SparseConvNeXt block is a full (not depthwise) 3×3×3 submanifold conv C→C, then LayerNorm, then a 4× MLP with SiLU.
     - 4 upsample blocks, each:
       - conv C→Cout·8
       - `Channel2Spatial(2)`
       - a `to_subdiv` linear whose 8 logits decide which child voxels exist (logit > 0).
     - Output: 7 channels per voxel at about the 1024 grid: dual-vertex offset (3, `2·sigmoid−0.5`), edge-intersect flags (3), quad split weight (1, softplus).
     - Mesh extraction is `o_voxel.flexible_dual_grid_to_mesh`: a hashmap lookup of 4 neighbours per edge to emit quads, split into triangles on the diagonal. This is not FlexiCubes and not marching cubes.
   - **Texture:** `SparseUnetVaeDecoder`, the same network without `to_subdiv`.
     - It reuses the shape decoder's child-voxel decisions (`guide_subs`), so its voxel tree is identical. Cache those gather maps and replay them.
     - Output: 6-channel PBR (base colour 3, metallic, roughness, alpha) via `*0.5+0.5`.
7. **GLB (`o_voxel.postprocess.to_glb`, CuMesh + nvdiffrast).**
   1. Fill holes and build a BVH.
   2. Narrow-band unsigned-distance dual-contouring remesh.
   3. QEM decimation.
   4. UV unwrap: GPU cone clustering plus xatlas per cluster.
   5. nvdiffrast rasterizes the mesh in UV space.
   6. Each texel is snapped to the original surface with the BVH and trilinearly samples the voxel attributes (`flex_gemm.grid_sample_3d`).
   7. Fill seams with `cv2.inpaint` (Telea).
   8. Write the GLB with trimesh, WebP textures and axis swap (y,z)→(z,−y).

## 3. Hard ops and what trellis.cpp does

| op | reference | trellis.cpp |
|---|---|---|
| Submanifold sparse conv 3×3×3 (both decoders, about 32 convs each; full 1024×1024×27 at the top stage) | `flex_gemm` (Triton); weights stored [Co,3,3,3,Ci] | Hashmap of coord→row, a [N,27] neighbour map cached per scale, missing neighbours point at a zero row, then 27 × (`get_rows` + matmul). Matches torch to 1e-3. |
| Channel→spatial upsample with child prediction | pure torch | Host prefix-sum to place children; the shape decoder's tree is cached and replayed for the texture decoder. Exact match. |
| Sparse attention, up to 49k tokens | flash_attn varlen (xformers optional) | Batch size 1, so it is plain dense flash attention. K/V are zero-padded to a multiple of 256 and kept in BF16; F16 overflows at the second-pass token counts. |
| 3D interleaved RoPE | complex view | Host cos/sin tables built from coordinates. |
| Voxel hashing | `o_voxel _C.hashmap_*_cuda` | CPU hashmap. |
| Mesh extraction | `o_voxel` | CPU, O(N). |
| Remesh / QEM / hole fill | CuMesh (CUDA) | CPU dual contouring; QEM on CPU/CUDA/HIP/Vulkan (`decimate_qem.comp`). |
| UV unwrap | CuMesh + xatlas | Vendored xatlas per cluster, or `--box-uv`. |
| Texture rasterization | nvdiffrast | CPU UV rasterizer, BVH closest-point snap, trilinear sampling, Telea inpaint port. |
| BiRefNet deformable conv | torchvision | Custom kernels in CUDA, Vulkan and CPU. |

**How complete trellis.cpp is:**
- It runs the whole pipeline end to end and produces a textured GLB with WebP PBR textures, which its authors call "at parity" with the reference postprocess.
- Backends: CUDA, ROCm/HIP, Vulkan, Metal, CPU and Qualcomm HTP.
- It has no SYCL wiring in its CMake. A Vulkan build on Arc should be the realistic way to run it here, but nobody has tested that.
- It reads its own ilintar GGUFs, not the Aero-Ex ones.
- Its noise comes from `std::mt19937`, so its outputs are not bit-identical to torch. For parity, inject dumped noise.
- Measured agreement with torch: SS sampler 4.3e-3 (exact voxel match), DiT 2.8e-3, DINOv3 1.8e-2, BiRefNet 4e-4.

The best porting reference is its per-component spec in `docs/spec/` (00–29) — note `00-synthesis.md` §"Hardest parts" — and its `src/sparse.cpp`, `dit.cpp`, `shape_decoder.cpp`, `dual_grid.cpp` and `uv_bake.cpp`.

## 4. Defaults and performance

- **Paths:** `1024_cascade` is the default; the others are `512`, `1024` and `1536_cascade` (1536 still uses the 1024 flows).
- **Seed:** 42.
- **Samplers:** all three are 12-step rectified-flow Euler with guidance applied only inside a time interval. Time is warped as `t' = r·t/(1+(r−1)t)`, then x −= Δt·v. The guidance rescale matches the std of x0.

| sampler | guidance | rescale | interval | rescale_t |
|---|---|---|---|---|
| sparse structure | 7.5 | 0.7 | [0.6, 1] | 5 |
| shape | 7.5 | 0.5 | [0.6, 1] | 3 |
| texture | 1.0 | 0 | [0.6, 0.9] | 3 |

- **Example GLB settings:** `decimation_target` 1M, `texture_size` 4096, `remesh=True`.
- **trellis.cpp GLB defaults:** 300K faces (150K at res 512); atlas 2048 (1024 at res 512).

**Runtime and memory:**
- H100: 512³ about 3 s, 1024³ about 17 s, 1536³ about 60 s. The reference needs 24 GB of GPU memory.
- trellis.cpp fits the 1024 cascade in 16 GB.

| | goblin (17k second-pass tokens) | turret (~45k tokens) |
|---|---|---|
| trellis.cpp, RTX 5060 Ti | 3:16 | 7:23 |
| trellis.cpp, Strix Halo Vulkan | 6:09 | 12:39 |
| reference, same 5060 Ti | 2:12 (3:37 cold) | 3:54 (5:20 cold) |

**Strix Halo breakdown for the goblin:**

| stage | time |
|---|---|
| DINOv3 | 25 s |
| sparse-structure flow (22 forwards) | 47 s |
| shape first pass | 28 s |
| shape second pass (8.3 s per forward) | 166 s |
| texture flow | 91 s |
| decoders | 20 s |
| postprocess | 24 s |

M5 at res 512: 9:21 with 5.6 GB peak RSS. The second-pass attention and the full-width top-stage sparse convs are where the time goes.

## 5. Reference path on Linux without CUDA

1. Clone `https://github.com/microsoft/TRELLIS.2` and use the Aero-Ex safetensors (or `TRELLIS.2-4B`) with CPU torch or Intel XPU torch.
2. Set `ATTN_BACKEND=sdpa`. Sparse attention only supports xformers or flash_attn, so monkeypatch it to SDPA, which is correct at batch size 1. trellis.cpp's `tools/ref_slat_shape.py` already does this.
3. **Sparse conv:** `flex_gemm` is pure Triton (https://github.com/JeffreyXiang/FlexGEMM) and might run on XPU through Intel's Triton backend; this is unverified. The fallback is the pure-torch gather conv in `tools/ref_sparse_conv.py`.
4. Replace the CUDA hashmap in `o_voxel` (`flexible_dual_grid_to_mesh`) with torch `unique`/`searchsorted`.
5. For parity, stop at mesh + voxel attributes. `to_glb` needs CuMesh and nvdiffrast, which are CUDA-only. Compare the GLB stage against trellis.cpp's `post-replay`, which reads a `TRELLIS_DUMP_POST` dump.
6. Dump `torch.randn` noise for each stage and feed it to the port.
7. Ready-made per-stage dumpers are in trellis.cpp `tools/ref_{birefnet,dinov3,ss_flow,ss_sample,ss_dec,ss_full,slat_shape,shape_dec,sparse_conv,c2s,deform}.py`. They have hardcoded paths and `DEV="cuda:1"`; edit both.

**URLs:**
- https://github.com/pwilkin/trellis.cpp
- https://github.com/microsoft/TRELLIS.2
- https://github.com/Aero-Ex/ComfyUI-Trellis2-GGUF
- https://huggingface.co/ilintar/trellis2-gguf
- https://github.com/JeffreyXiang/CuMesh