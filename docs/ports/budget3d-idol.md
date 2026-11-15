# Budget3D-Idol porting brief

Budget3D-Idol and Budget3D-Idol-Turbo (BudgetAny, Apache-2.0 weights, MIT code) turn one picture into 3D Gaussian
splats. They are fine-tunes of Sperid Labs' Idol / Idol-turbo (Apache-2.0) that run Idol's denoiser at a **token
budget**. Engine: `model3d/budget3d` (arch `budget3d-idol`), kernels `kernels/model3d/budget3d`, cards
`model-cards/model3d/budget3d-idol{,-turbo}.yml`, reference `reference/budget3d/`.

Sources checked:
- `BudgetAny/Budget3D` @9cfcb673 (scripts/sample.py, budget3d/octree.py, models/{denoiser,adapters,turbo,load,encoders}.py,
  sampling/{sampler,turbo}.py)
- `speridlabs/idol-sdk` @8db70034 (idol/models/{dit,layers,decoder}.py, conditioning.py, preprocess.py, gaussian.py, sampler.py)
- HF: `BudgetAny/Budget3D-Idol` @96f1493a, `BudgetAny/Budget3D-Idol-Turbo` @5d52410b, `speridlabs/idol` @2731c229

## 1. Files

| file | size | what |
|---|---|---|
| `BudgetAny/Budget3D-Idol/budget3d_idol.safetensors` | 1,512,422,068 | the denoiser, F32 EMA, 641 tensors (Idol's + `bp_*` adapters) |
| `BudgetAny/Budget3D-Idol-Turbo/budget3d_idol_turbo.safetensors` | 1,629,908,908 | the same plus Idol-turbo's LoRAs (`*.base.weight`, `slow_A/B`, `fast_A/B`): self-contained |
| `speridlabs/idol/denoiser.safetensors` | 1,482,258,308 | Idol's denoiser: only `pos_pe` [1, 8192, 3] is read (the Sobol token positions; Budget3D's files lack it) |
| `speridlabs/idol/decoder.safetensors` | 203,171,696 | OctreeProbabilityFixedlenDecoder (4 blocks) |
| `speridlabs/idol/decoder_gs.safetensors` | 949,083,036 | GaussianFixedlenDecoder (16 blocks) |
| `speridlabs/idol/birefnet.safetensors` | 444,466,856 | BiRefNet: tensor for tensor ZhengPeng7/BiRefNet's (max diff 0) minus 12 training heads - the `birefnet` key nextsycl already has is used |
| `facebook/dinov3-vith16plus-pretrain-lvd1689m/model.safetensors` | 3,362,432,800 | **gated** (manual approval, DINOv3 License) |
| `black-forest-labs/FLUX.2-dev/vae/diffusion_pytorch_model.safetensors` | 336,213,556 | **gated** (FLUX [dev] Non-Commercial License) |

The gated encoders are the cards' `local` parts: `NS_M3D_B3D_DINO` / `NS_M3D_B3D_VAE` in nextsycl.conf name them.

## 2. Pipeline

0. **Framing** (idol/preprocess.py): short side to 1024 (Lanczos); BiRefNet's alpha unless the picture has a real
   alpha (its input bilinear with corners aligned, the sigmoid back the same way, truncated to bytes); the alpha eroded
   (3 x 3 minimum); a square of 1.2 (synthetic) / 1.7 (real) x the alpha's bounding box; 1024 (Lanczos, premultiplied as
   PIL resizes RGBA); composited on black.
1. **Encoders.** DINOv3 ViT-H+/16 (1280 wide, 32 layers, 20 heads, SwiGLU 5120, 4 registers, RoPE base 100 on the
   patches) at 1024: 4101 tokens, `last_hidden_state` (the model's norm) then a parameter-free layer norm -> feature1
   [4101, 1280]. FLUX.2 VAE encoder, the posterior **mean** (Budget3D's deterministic encoder; Idol samples), 32
   channels at 128 x 128, patchified 2 x 2 (channel c 4 + dy 2 + dx), normalized by its BatchNorm statistics (eps
   1e-4), 5 zero rows in front -> feature2 [4101, 128].
2. **Denoiser** (LatentSeqMMFlowModel + Budget3D): 1024 wide, 16 heads of 64, MLP 4096 GELU-tanh. The latent [8192,
   16] -> input_layer + sinusoids of the Sobol positions; 2 noise-refiner blocks (time-modulated); the picture ->
   cond_embedder(feature1) + cond_embedder2(feature2) -> 2 context-refiner blocks (affine norms, no time; made once an
   image); the camera [5] -> a 2-layer MLP; 24 joint blocks over [latent | picture | camera]; the head (layer norm,
   shift/scale from the time) -> out_layer (16) and cam_out_layer (5). Every block's q and k are rotated by **RePo**,
   learned rotary positions (the block input -> LN -> SiLU-gated MLP -> a 3-D offset a head; angles pi x offset x 32
   learned frequencies, 10/10/12 pairs for x/y/z) and then per-head RMS-normed.
3. **Token budget** (budget3d/octree.py): the 8192 Sobol points are exactly two a 16^3 cell, so they form an octree
   (point / cell of 2 / cube of 8 cells). Detail scores log2(1 + sqrt(mean_c var) / 0.08) over a cell's 2 and a cube's
   16 points; the budget N = 512 + 7 k4 + k2; a bisected threshold offset decides k4, the top-k4 cubes open, the
   top-k2 of their cells open. Tokens [cubes | cells | points]: cubes and cells from ScaleMixer3D (mean of the children
   + a learned mix) plus an integrated positional encoding and a scale embedding. After the head, RefineStage (2
   AdaLN-zero blocks of 128 over a parent's children, 4 heads) turns each merged token's velocity into its children's,
   plus the Wiener gain t / (t^2 + 0.0625 (1 - t)^2) times z_t's residual inside the merged region. Budget 1 is
   exactly Idol's computation.
4. **Samplers.** Base: 50 Euler steps, time grid shifted by 3, CFG 3 against the zero condition (both branches of the
   joint blocks, the prefix once), the layout built from the noise at step 0 and refreshed at step 5 from the clean
   estimate. Turbo: 5 steps, no guidance, the phase expert by t (slow t >= 0.5), the context refined once per phase,
   the layout rebuilt every step.
5. **Decoders.** Octree: 8 levels from the root, each cell's 8 children logits (cross-only blocks over the latent,
   modulated by the level 2^l through a 2 pi sinusoid), systematic resampling of the cell's points among them, the
   leaves' points uniform in their 1/256 cells -> anchors (gaussians / 32). Gaussian decoder: 16 blocks (self over the
   anchors, cross to the latent, MLP) -> 480 an anchor = 32 Gaussians (offset tanh x 0.75 x softplus(offset scale),
   colour, scale softplus(+ inv_softplus(0.004)) with a 0.0009 floor, rotation x 0.1 + (1, 0, 0, 0), opacity
   sigmoid(+ logit 0.1)); z-up in [-0.5, 0.5]^3, written y-up as Idol's save_ply.

## 3. The port

- Kernels (`kernels/model3d/budget3d/b3d.cpp`, `mm.cpp`): layer norm with a modulation row per sequence, gated adds
  with a gate row per sequence, RePo's rotation + per-head RMS norm (a sub-group a head), the refiners' attention
  within <= 16 children, gathers with an added table (token assembly), ScaleMixer's per-channel mixing, group mean,
  the refiner's input, and the VAE's group norm, padding and softmax; float32-output products on oneDNN. The shared
  diffusion kernels do the linears, attention (`nsd_attention_qk`), layer / RMS norms, GELU, SwiGLU, RoPE, conv2d / 3d.
- The guided branches share one batch through the joint blocks (rows [x | picture | camera] twice). RePo, the
  refiners, the heads and the mixers run in float32 (as Budget3D keeps them); the blocks half with a float32 stream.
- Idol-turbo's LoRAs are merged at load (W + 4 B A on the GPU in float32, then half): two copies of the adapted matrices.
- The noise is PyTorch's (`torch.manual_seed` then `randn` of the latent and of the camera's 5 values - the scalar
  path, doubles, the second of each pair cached), so a seed gives the reference's starting point.
- On the host: the time / level embeddings, the camera MLP and head row, the layouts, the refiners' Wiener term and
  the scatter back to Idol's order, the Euler steps, the octree's resampling, the Gaussians' activation, the PLY.
- Splats are a new result of the 3D contract (`nextsycl_model3d::Splats`: PLY in and out, a CPU rasterizer for checks
  and `nextsycl model3d tool render`), `Abilities::splats` / `Model3dEngine::splat`; the 3D server writes `.ply`
  (response_format ply or url), the front door serves and lists them (`does: [generate, splat]`: capabilities splats,
  ply), and wfe/model3d draws them (`splat.ts`: EWA quads, back-to-front after a 16-bit counting sort) with a token
  budget slider (options.budget).
