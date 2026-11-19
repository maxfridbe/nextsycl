# Hi3D v1.0 on nextsycl

[← 3D](../3d.md)

Hi3D v1.0 (hi3d-official, 2026-10-23; built on Sparc3D, arXiv:2505.14521) refines a **coarse mesh** into a
high-resolution, geometry-only mesh, conditioned on a **picture of the same object**. Upstream is CUDA-only (sm80+,
flash-attn, spconv, Triton) and its 1536³ ladder was validated only on a 96 GB H20. Here it is the `model3d/hi3d`
engine (Rust host, SYCL kernels in `kernels/model3d/hi3d` and its silo `kernels/model3d/hi3d/silo`), cards
`hi3d-1.0` (512 / 1024 / 1536) and `hi3d-1.0-512` (one pass; no `transformer_1024`).

Sources: github.com/hi3d-official/Hi3Dv1.0 @13657ad (the `sparc3d` package), huggingface.co/hi3d-official/Hi3Dv1.0
@23d1bace, facebook/dinov2-large @47b73eef, openai/clip-vit-large-patch14 @32bd6428.

## Use

```sh
nextsycl models pull hi3d-1.0                       # 9.3 GB of Hi3D files + DINOv2-large 1.2 GB + CLIP-L 1.7 GB (+ BiRefNet)
nextsycl model3d gen picture.png --model hunyuan3d-2mini-turbo --out coarse.glb       # a coarse mesh in seconds
nextsycl model3d refine picture.png --mesh coarse.glb --model hi3d-1.0 [--resolution 512|1024|1536] [--seed 0] [--steps 25] [--faces N]
nextsycl model3d check DUMP --model hi3d-1.0 --stages pre,cond,vox,dit,steps,vae,mesh,post [--structure 64|128] --picture P --mesh coarse.glb
```

Front door: **`model3d.refine`** `{image, mesh | file, resolution, seed, steps, faces, options: {euler}}` - a job like
`model3d.generate`. With no `mesh` / `file` it first makes the coarse mesh with a generator from the same picture and
seed (`coarse_model`, default `hunyuan3d-2mini-turbo` when enabled) and links it in the answer as `coarse`; the 3D
servers share one output directory, so the refiner reads that GLB by name. `model3d.generate` naming a refining model
with a `mesh` / `file` does the same refinement (the contract's `Abilities::refine`). Cards say `does: [refine]`;
`system.models` lists the capability `refine` and the method `model3d.refine`.

## The pipeline (as `Sparc3DPipeline.__call__`)

1. **Picture** (`prep.rs`): the alpha's bounding box squared (PIL getbbox / crop, transparent past the edges),
   pasted on white with PIL's 8-bit rounding, padded to 1.2x its side. DINOv2's input: torch bilinear to 756
   (align_corners false, no antialias); CLIP's: PIL Lanczos to 224 (8-bit). An opaque picture is cut out by the shared
   BiRefNet first unless `remove_background: false` (upstream: opt-in; nextsycl's 3D default).
2. **Condition** (`vision.rs`): DINOv2-large's last hidden state (54 x 54 patches + CLS = 2,917 tokens; the 37 x 37
   position table resized bicubically, A = -0.75, as transformers 4.46.3) and CLIP ViT-L/14's pooler output, each
   L2-normed; CLIP's 1,024 beside every DINOv2 token: [2917, 2048], computed once for all passes.
3. **Voxels** (`vox.rs`): the coarse mesh centred on its eps-expanded box, scaled to the grid less a 3-voxel margin;
   every voxel each triangle's box covers tested by the Triton kernel's separating-axis test (float32), the hits
   unique by `z << 22 | y << 11 | x` in that order (the order matters: the noise rows follow it); shrunk 5 % at a time
   until <= 80,000 (64³), 90,000 (128³), 110,000 (192³).
4. **Flow** (`dit.rs`, `sched.rs`): noise from PyTorch's CPU generator at the seed (one stream across the passes:
   `nextsycl_diffusion::torch_rng`, value for value). Sparc3DTransformer: input_layer 32 -> 256, ResBlock(256),
   ResBlock(256 -> 2112) with the paired downsample by 2 (the parents' mean **with the zero row counted**:
   `scatter_reduce(mean)`'s default `include_self=True` divides by count + 1 - missing it put the first check at cos
   0.98), 24 blocks of 2112 (22 heads of 96: self-attention with the rotary embedding of the parents' coordinates
   **then** the per-head RMS norm on q and k, cross-attention to the 2,917 tokens, GELU-tanh MLP of 8448; adaLN shared +
   each block's own), the upsampled skips, layer norm, 32 out. 25 Heun updates over the token-warped schedule
   (sigma_min 0.002, rational rescale 2, then the token warp clamp((19n - 20000)/(3n + 60000), 1, 5); sigma inflated 4 %
   above warped 0.985, 0.2 % below); guidance against the all-zero condition only at model times >= 0.7 (4 at the
   first, 10 t² on [0.7, 1)), each Heun slope at its own time. The zero condition's cross-attention is the value
   bias (keys all equal): one row a block, no attention run. Then x * latent_std + latent_mean.
5. **Decoder** (`vae.rs`): z_to_f conv 32 -> 512; mid: ResBlock, two serialized-attention blocks (conv position
   encoding, windows of 48 along z-order then transposed z-order, the last window the last 48 voxels as upstream's
   padding makes it), ResBlock; dec3 / dec2 / dec1 each 2 ResBlocks then an unpooling (proj C -> 8C, occupancy head
   GroupNorm+Swish+conv C -> 8, children with logit > 0 kept at 2x + offset); dec0; GroupNorm, Swish, conv -> 8 sign
   logits + 32 attributes (tanh). PointGroupNorm is one mean / variance a channel group over every voxel (float32
   E[x²] - mean²; block sums on the GPU, added in double on the host). The children's neighbour maps come from their
   parents' on the GPU (no hashing past the latent level). ResBlocks above 12M voxels pass through (upstream's guard).
6. **Mesh** (`extract.rs`): voxels sorted by (x, y, z); per corner the sign probability, distance, offset x 0.5
   averaged over the voxels sharing it; sign from the averaged probability; the corners mapped by the latent grid's
   matrix (centred on the latents' mean, scaled to their radius - applied to the 8x finer coordinates, as upstream);
   FlexiCubes without learned weights (sparc3d's sparse ambiguity lookup), then every boundary loop of 3..10,000
   edges closed by ear clipping (float32, numpy's argmax ties).
7. **Next pass / finish**: the mesh voxelized at the next structure grid; at the end pieces under 0.5 % of the faces
   dropped (trimesh's face adjacency: edges of exactly two faces), decimated to 1M faces (2M at 1536), centred on the
   vertex mean, radius 1. The decimation is the mesh crate's slab-parallel quadric collapse, not pyfqmr.

Precision: fp16 weights as stored (the transformers' torso; the decoder's and encoders' float32 rounded to half
once), half activations into the products with float32 accumulation and float32 residual streams; the input and output
layers, timestep MLP and modulations in float32. The 96-wide heads are zero-padded to 128 (q scaled by sqrt(128/96))
so the attention runs on the flash kernel (libnextsycl-flash, D 64 / 128 only); the padding is exact in arithmetic.

## Parity (exact before fast)

`reference/hi3d/ref.py` runs upstream's own modules on the CPU in float32 (python -I, its own directory), on the same
files, with stand-ins for the CUDA parts only: flash-attn's varlen functions as SDPA (the decoder's `qkv.half()` kept),
spconv's SubMConv3d as a 27-tap gather (tap (a, b, c) of [Co, 3, 3, 3, Ci] reads (x + a - 1, ...)), the Triton
voxelizer as the same SAT in float32 numpy. The transformers run in float32 (their fp16 weights widened exactly).
Input: upstream's assets/example.png (a dragon bust, RGBA) and a coarse mesh from our hunyuan3d-2mini-turbo (seed 0,
40,000 triangles); seed 0. Each stage below starts from the reference's own input (`nextsycl model3d check`, B65).

| stage | rel RMS | cos | notes |
|---|---:|---:|---|
| DINOv2 input / CLIP input | 2.1e-8 / 0 | 1.0 | the framing and both resizes bit-for-bit (CLIP's) |
| DINOv2-large last hidden state | 1.5e-3 | 0.999999 | half products |
| CLIP pooled | 4.0e-3 | 0.999992 | |
| condition [2917, 2048] (own picture) | 3.1e-3 | 0.999995 | |
| voxels at 64³ | identical | | 12,961 voxels, same order |
| schedule (sigmas, inflated times) | 0 | | |
| transformer_512, first call (2,899 tokens) | in0 6.5e-4, in1 4.9e-4, blk0 2.5e-4, blk23 1.5e-3, out1 6.9e-4, **vel 7.3e-4** | 1.0 | zero condition 7.4e-4 |
| transformer_1024, first call (63,454 voxels, 14,689 tokens) | in1 5.8e-4, blk23 5.5e-4, **vel 7.3e-4** | 1.0 | zero condition 7.2e-4 |
| 25 Heun updates (pass 1) | after 1: 2.7e-4, 10: 9.2e-4, 20: 4.0e-3, 25: 7.8e-3; **latents 8.4e-3** | 0.99997 | the error grows through the guided steps, as in the other ports |
| decoder z_to_f / mid | 2.8e-4 / 3.3e-4 | 1.0 | at the reference's latents |
| unpooling logits (3 levels) | 1.8e-4 / 1.9e-4 / 2.0e-4 | 1.0 | children decided otherwise: 1 of 103,688; 17 of 458,656; 99 of 1,944,344 (logits near 0) |
| final sign logits / attributes | 1.7e-4 / 4.5e-4 | 1.0 | at the reference's pruning |
| decoder on its own pruning | | | 983,152 voxels against 983,143 (212 only here, 203 only there) |
| FlexiCubes (reference's decoded voxels) | | | **966,470 vertices, 1,896,818 faces: identical counts**; vertex Chamfer 4.5e-7 |
| holes filled | | | **1,932,064 faces: identical** |
| postprocess (reference's filled mesh) | | | 499,540 v / 999,999 f both; Chamfer 7.4e-3 in unit-radius units with each side's own centring, **4.7e-4 of the box side with both box-normalized** (the decimators differ: pyfqmr against the mesh crate's) |
| whole run, 512 (picture + coarse mesh -> final) | | | Chamfer against the reference's final 4.7e-4 of the box side (box-normalized), the same as postprocess alone: the earlier stages add nothing visible |

The second pass's reference (`--structure 128`: voxels 63,454 at 128³ from the reference's pass-1 mesh, identical
to ours from the same mesh) runs ~6 min an update on the CPU; its transformer check is above; see "Unverified" for
the rest.

## Speed and memory

`nextsycl model3d refine` of the example (seed 0, 25 Heun updates a pass, `NS_M3D_HI3D_PROFILE=1`: synced stage
timings; models load stage by stage; peak = the card's most memory in use, every process's, watched every 50 ms):

| card | resolution | condition | pass 1 sampling (2,899 tokens) | pass 2 sampling (14.7k tokens) | decode | mesh (FlexiCubes + holes) | postprocess | total | peak |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B70 | 512 | 3.4 s | 25.5 s | - | 1.5 s (983k voxels) | 3.2 s (1.93M faces) | 5.8 s | **39.4 s** | 5.1 GiB |
| B70 | 1024 | 3.3 s | 25.3 s | 173.5 s | 1.2 + 3.1 s (4.39M voxels) | 2.4 + 11.8 s (8.69M faces) | 12.2 s | **233.1 s** | 8.7 GiB |
| B65 | 512 | 3.1 s | 39.2 s | - | 1.6 s | 2.3 s | 4.7 s | **51.0 s** | 5.0 GiB |
| B65 | 1024 | 3.3 s | 38.9 s | 297.9 s | 1.6 + 4.0 s (4.39M voxels) | 3.1 + 11.5 s (8.69M faces) | 12.8 s | **373.3 s** | 8.6 GiB |

A pass-2 transformer call (14.7k tokens) on the B70 is ~2.0 s: the 24 blocks' products 0.91 s (~48 TFLOP/s of
useful work), self-attention 0.93 s, the I/O ResBlocks 0.16 s; a pass is 25 updates = ~90 calls (Heun's two slopes,
the zero condition above 0.7). On the B65 the same call is ~3.1 s. The padded flash attention saved ~10 % of the
self-attention against oneDNN's kernel at D = 96 (1.79 -> 1.61 s on the B65); the attention is compute-bound at these
token counts.

**1536** (B70, the same example): 673.5 s, peak **14.3 GiB** (of 32: room to spare with 1.5 GiB kept free). Pass 3's
voxels did not fit 110,000 at 192³: the grid settled at 164³ (108,879 voxels, 25,583 tokens; upstream does the same
and warns, keeping the 1536 label); its sampling 399 s, decode 7.46M voxels in 4.7 s, FlexiCubes + holes 14.7M faces
in 21 s, postprocess to 2M faces 24 s. Sizing before the run: the third pass's transformer at ~27k tokens needs well
under 2 GiB of activations; the decoder's finest level ~10M voxels x (128 float32 features + their half copy + a
float32 conv output + the neighbour map) ~ 14 GiB worst case plus the 1024-wide unpooling projection of its parents
(~5 GiB, freed before): measured 14.3 GiB at the peak.

The 3D server path was run too (`model3d serve hi3d-1.0-512 --here`, POST /v1/model3d/generate with image + mesh,
B65): 51.6 s, the GLB and its settings in the output directory; `model3d.info` reports `abilities.refine`.

## Licence

- Hi3D's eight files (and its code): **Hi3D Open License Agreement v1.0** (OCEANO HK HOLDING LIMITED). Commercial use is
  allowed; a Related Product or Service above **200,000 MAU or USD 1,000,000 monthly revenue** must notify the
  licensor and apply for a separate licence within 60 days (Section 5); commercial integrations display **"Powered by
  Hi3D 1.0"** linking http://hi3d.ai/ (Section 7; an API without a UI: in its customer documentation); distributing the
  weights carries the Agreement, the Section 8.2 NOTICE text and Schedule 1's Acceptable Use Policy; outputs (meshes)
  that contain no Hi3D materials carry none of these duties (Section 9); outputs may train other models (2.2).
  Hong Kong law, HKIAC arbitration.
- Third-party: TRELLIS / Pointcept / O-CNN code MIT, FlexiCubes (and its tables, copied into `tables.rs`) Apache-2.0;
  DINOv2-large Apache-2.0; CLIP ViT-L/14 MIT; BiRefNet MIT. A coarse mesh from Hunyuan3D-2mini keeps the Tencent
  Hunyuan 3D 2.0 Community License (not licensed in the EU, UK, South Korea). The cards carry this.
- upstream's assets/example.png is excluded from the Hi3D licence (Schedule 2): used here only as a local test input,
  not committed.

## Unverified / open

- The second pass's reference sampler, decoder and mesh (`--structure 128` steps / vae / mesh / post) and a 1024 whole-
  run Chamfer: PASS2_NOTE
- The final decimation is not pyfqmr's (counts match on the example; geometry within the vertex spacing after
  box normalization, but the centroid / radius normalization of a differently decimated mesh shifts it by ~7e-3).
- Only one picture / mesh pair was checked (the dragon bust). Opaque pictures through BiRefNet were not checked against
  upstream (upstream takes them whole unless asked).
- Speed: the flow is the time. Not done: a fused rope + RMS + head padding, caching the modulations across the
  conditional and zero-condition calls of one time, CUDA-graph-like replay, the transformer's products in int8.
