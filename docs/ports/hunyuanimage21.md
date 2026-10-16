## HunyuanImage 2.1: research report

I made no project file changes. I read the GGUF headers with 8 MB range requests and the source through raw GitHub. Reference sources are kept in `/tmp/claude-1000/-var-home-maxfridbe-Dev-hardac/ca6b4189-dc54-4ec0-aa62-40001a0681e3/scratchpad/hyref/`.

### 1. Files in calcuis/hunyuanimage-gguf

All DiT files use `general.architecture=pig` and ComfyUI tensor names with fused qkv. There are 856 tensors, and the large matrices use the stated quant.

- **Base DiT, `hunyuanimage2.1-*`**: q2_k_s 6.51, q2_k 7.32, q3_k_s 9.05, q3_k_m 9.75, q4_k_s 10.78, iq4_xs 10.77, iq4_nl / q4_k_m 11.30, q4_0 11.39, q4_1 12.36, q5_k_s 12.93, q5_0 / q5_k_m 13.42, q5_1 14.48, q6_k 15.68, q8_0 19.79, mxfp4_moe 18.96, plus `hunyuanimage2.1_fp32-f16.gguf` at 35.90 GB (sizes in GB).
  - In the q8_0 file the small tensors (token refiner, time_in, img_in, final) are F32 and byt5_in is BF16.
- **Base "v2", `hunyuanimage2.1-v2-*`**: same weights with the small tensors in F16. Quants run from q2_k 6.50 to q8_0 18.96 GB.
- **Distilled, `hunyuanimage2.1-distilled-{iq4_nl,q2_k,q8_0}`**: q8_0 is 19.83 GB.
  - The `-distilled-v2-*` set runs from q2_k 7.50 to q8_0 19.95 GB, and the README says "use v2".
  - It adds `guidance_in.*` and `time_r_in.*` (MeanFlow), for 864 tensors.
- **Refiner, `hunyuanimage-refiner-*`**: 3 files (q2_k, iq4_nl, q8_0 17.09 GB). The README says to use `-refiner-v2-*`, which runs from q2_k 5.64 to q8_0 17.21 GB.
  - It is a separate DiT with hidden size 3328, no byT5, and an `img_in` Conv3d with 128 input channels.
- **"lite" and "lite-v2.x"**: a calcuis-only variant with tensors renamed (`*_blk_qkv` instead of `*_attn_qkv`) and the same shapes. The README describes it as 8 steps at cfg 1. Not official; skip it.
- **VAEs**:
  - `pig_hunyuan_image_vae_fp32-f16.gguf`, 0.81 GB: the 32x 2D VAE.
  - `pig_hunyuan_image_refiner_vae_fp32-f16.gguf`, 2.52 GB: the refiner's 16x 3D VAE.
- **Text encoder**: `qwen2.5-vl-7b-test-q4_0.gguf`, 5.03 GB. It uses `arch=dog` and contains the LM plus the vision tower, the same layout as the Qwen-Image text encoder.
- **Workflows and demos**: 4 workflow JSONs and 3 demo PNGs.
- **byT5 is in a different repo**, calcuis/pig-encoder: `byt5_small_glyphxl_fp32-{f16 439 MB, q8_0 235, q5_0 154, q4_0 / iq4_nl 127 MB}`.

The Civitai version 2201762 is not a model. It is `hunyuanimage21TextToImage_v10.zip`, a 2 KB ComfyUI workflow:
- Files: hunyuanimage2.1 q8_0, qwen2.5-vl-7b q4_0, byt5 f16, and the pig VAE.
- Settings: 1280x704, 20 steps, CFG 2.5, Euler.

**Recommended set, about 26.1 GB total:**

| Card | File | Size |
|---|---|---|
| B70 | https://huggingface.co/calcuis/hunyuanimage-gguf/resolve/main/hunyuanimage2.1-q8_0.gguf | 19.79 GB |
| B70 | https://huggingface.co/calcuis/hunyuanimage-gguf/resolve/main/pig_hunyuan_image_vae_fp32-f16.gguf | 0.81 GB |
| B65 | https://huggingface.co/calcuis/hunyuanimage-gguf/resolve/main/qwen2.5-vl-7b-test-q4_0.gguf | 5.03 GB |
| B65 | https://huggingface.co/calcuis/pig-encoder/resolve/main/byt5_small_glyphxl_fp32-f16.gguf | 0.44 GB |

- For the fast path, add `hunyuanimage2.1-distilled-v2-q8_0.gguf` (19.95 GB). Load it instead of the base DiT, not alongside it.
- If a Qwen2.5-VL-7B GGUF from the Qwen-Image engine is already on disk, reuse it and skip the 5 GB download.
- The refiner (17.2 + 2.5 GB) cannot share the B70 with the base DiT, so it would have to be loaded in turn. Defer it.

### 2. Architecture

Source: `hunyuanimage_config.py` and `hunyuanimage_dit.py` in the official repo, confirmed against the GGUF shapes.

**DiT (about 17B)**
- 20 double-stream blocks and 40 single-stream blocks.
- Hidden size 3584, 28 heads of 128, MLP ratio 4 (14336), GELU-tanh activation.
- Q/K RMSNorm over the 128-wide head, with biases on qkv, proj and MLP.
- Patch size [1,1]: `img_in` is a 1x1 conv from 64 to 3584. Input is 64 channels and output is 64.
- Double block: img/txt modulation with 6x3584 outputs, separate weights per stream, and joint attention over the concatenation [img, txt].
- Single block:
  - `linear1` maps 3584 to 25088 (qkv 10752 plus MLP 14336).
  - `linear2` maps 17920 (attention output plus GELU(MLP) output) back to 3584.
  - 3-way modulation.
- `final_layer`: adaLN with 2x3584, then a linear to 64.
- RoPE:
  - 2D, `rope_dim_list=[64,64]` for (h, w), theta 256.
  - Grid starts at 0 and is not centred.
  - Pairs are interleaved (GPT-J style).
  - Applied to image tokens only; text tokens get no RoPE.
- Token refiner (`txt_in`, SingleTokenRefiner):
  - `input_embedder` linear from 3584 to 3584.
  - 2 blocks, each: LayerNorm with affine, then self-attention with the text mask, then gated MLP. The adaLN input is `t_embedder(t)` plus `c_embedder(masked mean of the text)`.
- Timestep embedding: sinusoidal with 256 dims, cos then sin, max period 10000, followed by an MLP.
- Distilled model:
  - `guidance_embed`: `vec += guidance_in(g*1000)`.
  - MeanFlow: `vec = (time_in(t) + time_r_in(t_next)) / 2`, where `t_next` is the next timestep and 0 on the last step.
  - No CFG batch is run.
- The text/byT5 order inside the model, `reorder_txt_token`, is described under byT5 below.

**Text conditioning**
- Qwen2.5-VL-7B, `hidden_state_skip_layer=2`, which is `hidden_states[-3]` (ComfyUI `layer_idx=-3`), with no final norm.
- Template: `<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>`.
- `crop_start=34`, maximum 1000 tokens.
- Default negative prompt: "Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion".

**byT5 glyph path**
- Quoted text is extracted with the regexes for `"…"`, `‘…’` and `“…”`.
- It is formatted as `Text "{t}". ` and byte-tokenized: ByT5 plus 1126 added color/font tokens, vocab 1510, max length 128.
- Encoder: T5 with 12 layers, d=1472, 6 heads, relative bias with 32 buckets, gated-GELU FFN 3584.
- It then goes through `byt5_in` (LayerNorm 1472, fc1 to 2048, GELU, fc2 to 2048, GELU, fc3 to 3584).
- With no quotes, the input is zeros with a zero mask.
- Inside the model, `reorder_txt_token` concatenates the sequence as [valid byT5, valid Qwen, padded byT5, padded Qwen].

**VAE (HunyuanVAE2D)**
- 64 latent channels, 32x spatial compression, so 2048² gives a 64x64 latent (4096 tokens).
- Scaling factor 0.75289, no shift.
- Decoder:
  - `conv_in` from 64 to 1024, plus a shortcut that repeat-interleaves the input channels.
  - Mid section: resblock, single-head attention, resblock.
  - 6 levels with channels 1024, 1024, 512, 512, 256, 128 and 3 resblocks each, using GroupNorm(32) and swish.
  - Upsample is DC-AE style: conv to 4C', pixel shuffle 2x2, plus a shortcut of repeat_interleave and pixel shuffle.
  - Then norm, swish and `conv_out` to 3 channels.
- The encoder mirrors this with space-to-depth and a channel-mean shortcut.

**Refiner**
- Separate DiT: 20 double and 40 single blocks, hidden 3328, 26 heads, 3D RoPE [16,56,56], in_channels 128 (noisy latent concatenated with the condition latent), `guidance_embed` plus MeanFlow.
- Uses the 3D causal Conv3d VAE: 32 channels, 16x spatial, 4x temporal, RMS-gamma norms, scaling 1.03682.
- Older template `dit-llm-encode` (Llama-style headers), crop 36, max 256 tokens.
- Runs 4 steps.

**Defaults** (from `hunyuanimage_pipeline.py`)

| | Steps | Guidance | Shift | Notes |
|---|---|---|---|---|
| Base | 50 | 3.5 | 5 | CFG mode "MIX_mode_0", i.e. APG |
| Distilled | 8 | 3.25 (embedded) | 4 | no negative prompt |

- APG settings: rescale 10, momentum -0.5, eta 0. It switches on after step 5, or after step 38 when quoted text is present.
- Sigma schedule: `s = shift*s / (1 + (shift-1)*s)` over `linspace(1, 0, N+1)`, with timestep = 1000·s.
- Sampler: plain Euler, `x -= (s_i - s_{i+1}) * v`.
- Native resolution 2048x2048; width and height must be divisible by 32.
- Community runs: calcuis reports 12-15 steps; the Civitai workflow uses CFG 2.5 with Euler.

### 3. Reference implementations

- **Official repo**, https://github.com/Tencent-Hunyuan/HunyuanImage-2.1, files under `hyimage/`:
  - `models/hunyuan/modules/{hunyuanimage_dit,models,token_refiner,posemb_layers,embed_layers}.py`
  - `models/vae/hunyuanimage_vae.py` (27 KB, self-contained) and `refiner_vae.py`
  - `models/text_encoder/{__init__,byT5/__init__}.py`
  - `diffusion/pipelines/hunyuanimage_pipeline.py`
  - Official weights: https://huggingface.co/tencent/HunyuanImage-2.1 (bf16 DiT 34.9 GB).
- **ComfyUI** added the regular model on 2025-09-10 (PR #9792) and the refiner plus refiner VAE on 2025-09-12 (#9817):
  - `comfy/ldm/hunyuan_video/model.py`, which is shared with HunyuanVideo
  - `comfy/ldm/hunyuan_video/vae.py` and `vae_refiner.py`
  - `comfy/text_encoders/hunyuan_image.py`, `byt5_config_small_glyph.json` and `byt5_tokenizer/`
  - ComfyUI's tensor names match these GGUFs (fused qkv).
- **Diffusers** has `HunyuanImagePipeline` and `HunyuanImageRefinerPipeline` (PR #12333, 2025-10-24) in `src/diffusers/pipelines/hunyuan_image/`. Converted repos: `hunyuanvideo-community/HunyuanImage-2.1-{,Distilled-,Refiner-}Diffusers`.
- **Most compact reference for CPU parity dumps:**
  - Use ComfyUI `hunyuan_video/model.py` and `vae.py` for the DiT and VAE, because they load these GGUF names directly through the calcuis gguf node.
  - Use the official `hunyuanimage_pipeline.py` for the sampler and APG ground truth.
  - The official `hunyuanimage_vae.py` is the cleanest standalone VAE and is easy to run in fp32 on CPU.

### 4. Ops the Qwen-Image engine does not have yet

1. **Single-stream block**: fused `linear1` split into qkv and MLP, parallel attention and MLP, and `linear2` over the concatenation. Qwen-Image is double-stream all the way through.
2. **GELU-tanh MLP with biases everywhere** (Qwen-Image uses plain GELU), and a 3-way modulation for the single blocks.
3. **RoPE**: 2D interleaved-pair RoPE with theta 256, image tokens only, text unrotated. This differs from Qwen-Image's MSRoPE with positioned text.
4. **Token refiner**: LayerNorm with affine, masked self-attention over text only, a masked-mean pooled `c_embedder`, and its own `t_embedder`.
5. **Variable-length joint attention**: key-padding mask over the reordered [byT5 | Qwen] text, or trimming the padding.
6. **MeanFlow and guidance embedders** (distilled): averaged t/r embedding plus `guidance_in`.
7. **byT5-small encoder** (new T5 path): T5 RMS LayerNorm, bucketed relative attention bias, unscaled attention, gated-GELU FFN, a byte tokenizer with 1126 added tokens, quote extraction, plus the `byt5_in` MLP.
8. **Qwen2.5-VL output tap**: take hidden layer -3 with no final norm, and use crop 34 with the v2 template.
9. **VAE**:
   - The main 2D VAE uses GroupNorm(32) and swish instead of Wan's causal Conv3d and RMS norm.
   - It needs pixel shuffle, depth-to-space, a repeat_interleave channel shortcut, and single-head spatial attention at C=1024.
   - Large decodes need tiled blending; the official code ships tiling.
   - The refiner VAE is a 3D causal VAE, closer to Wan.
10. **APG guidance**: momentum buffer, norm-threshold rescale and projection, with separate thresholds for OCR prompts. This is only for the base model; plain CFG also works.
11. **Static-shift flow schedule**: probably a small change, since Qwen-Image uses a dynamic shift.