## MOSS-Image-SFT (OpenMOSS): port plan

Text to image, 5.1B, Apache-2.0, released 2026-10-16. Served here from unsloth's GGUFs
(`unsloth/MOSS-Image-SFT-GGUF`: the DiT only, Q3_K_XL ... BF16, 2.8 - 10.2 GB). Upstream:
`OpenMOSS-Team/MOSS-Image-Base` / `-SFT` (diffusers `MOSSImagePipeline`; the whole model is one file,
`pipeline_moss_image.py`, 440 lines - kept beside the reference as `reference/mossimage/pipeline_moss_image.py`).
Prompts are natural language, booru-style tags, or a JSON layout (a summary and items with bounding boxes: where
text and objects land, and what the text says).

### Files

| Part | File | Size |
|---|---|---|
| DiT | `unsloth/MOSS-Image-SFT-GGUF` `MOSS-Image-SFT-UD-Q4_K_XL.gguf` (recommended), `-Q8_0`, `-UD-Q5_K_XL`, `-UD-Q6_K_XL`, `-Q4_K_M`, `-UD-Q3_K_XL`, `-BF16` | 3.42 GB |
| text encoder | `unsloth/Qwen3.5-9B-GGUF` `Qwen3.5-9B-Q8_0.gguf` (the text model; no mmproj) | ~9.5 GB |
| VAE | `unsloth/MOSS-Image-SFT-FP8` `vae/diffusion_pytorch_model.safetensors` (FLUX.2 VAE, fp32) | 250 MB |
| tokenizer | the same repo's `tokenizer/` (Qwen2 tokenizer.json + chat_template.jinja) | 20 MB |

The GGUF's metadata is four keys (`general.architecture = moss_image`, ...); its tensors are the PyTorch state-dict
names (`layers.N.proj.q.weight`, ...), 463 of them, `ne` = [in, out]. Types: Q4_K / Q5_K / Q6_K / BF16 matrices, F32
norms and biases. `rope.bands` [16] is a derived constant (recomputed: `1 / 10000^(i/16)`).

### Architecture (from `pipeline_moss_image.py`)

- **Latents**: the FLUX.2 VAE's, packed 2x2: a 1024x1024 picture is 128 channels x 64 x 64 (patch 16 px). Sizes are
  multiples of 16. Noise is fp32.
- **DiT** (`MOSSImageTransformer`): dim 3072, 24 heads of 128, 42 blocks, ffn 8192, one stream: the image tokens
  (`video_embedder`: Linear 128 -> 3072) followed by the text tokens (`text_embedder`: Linear 4096 -> 3072 and two
  small text blocks with biases and fused qkv, run on the text alone), then 42 `_Block`s over both.
  - block: `x += o(attn(q,k,v of (rmsnorm1(x) * (1 + scale_a) + shift_a))) * gate_a`; `x += down(situ(up, gate)) *
    gate_f` with `rmsnorm2` and `(shift_f, scale_f)`. q / k have a per-head RMSNorm (128), no biases in the blocks.
  - the modulation is **shared**: `time -> shared_block_modulation (Linear 3072 -> 6 x 3072)` once a step, the same
    six vectors for every block (no per-block adaLN).
  - `situ(h, g) = 4 tanh(g/4) sigmoid(g) * 25 tanh(h/25)` (a soft-clipped SwiGLU); the FFN is one `ffn_up_gate`
    matrix (3072 -> 16384) split in two.
  - RoPE: 4 axes (r, f, h, w) over 16 bands (`head_dim / 8`): image tokens (0, text_len, h - (H-1)/2, w - (W-1)/2),
    text tokens (0, i, 0, 0); phase = coordinate x band, `[sin | cos]` per axis, applied half-split (`rotate_half`).
  - time: sinusoid (256, cos then sin, float64) -> Linear -> SiLU -> Linear; the head: `rmsnorm(x_img) * (1 + s) + t`
    from `video_head_modulation`, then Linear 3072 -> 128.
  - **skip layers**: a forward can skip whole blocks by index; the negative branch of CFG skips block 22 (that, not a
    negative prompt, is the default: `negative_prompt = ""`, `uncond_skip_layers = (22,)`).
- **Text encoder**: Qwen3.5-9B (`Qwen3_5Model` text side): 32 layers, hidden 4096, 3 gated-DeltaNet layers then 1 gated
  full-attention (16 heads, 4 kv, head 256, output gate, 25 % partial rotary, mrope-interleaved theta 1e7), FFN 12288.
  The pipeline feeds `apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
  enable_thinking=True)` and takes `last_hidden_state` (after the final norm) of **every** token: the DiT's text
  sequence is the whole prompt (up to 4096 tokens), unpadded, one prompt at a time. No LM head.
- **Sampler**: `MOSSImageScheduler`: a shifted grid (shift 5), snapped to the nearest of the 1000 training sigmas
  (`unique_consecutive`), Euler on the velocity: `x += (sigma_next - sigma) * v`. CFG 3 over fp32 predictions,
  50 steps default. Optional SLG (an extra skipped positive forward) - not needed at first.
- **VAE decode**: `z * sqrt(bn.running_var + 1e-4) + bn.running_mean`, unpatchify (128 -> 32 channels, 2x2),
  `AutoencoderKLFlux2.decode` (decoder channels 96/192/384/384, 2 resnets a block, mid attention, group norm 32,
  post_quant_conv), clamp [-1, 1]. VAE runs in fp32 upstream.

### What exists in the repo to reuse

| Need | Have |
|---|---|
| quantised GEMM Q4_K / Q5_K / Q6_K / Q8_0 / BF16, rmsnorm + modulation, rope tables, attention, gate_add, conv2d | `nextsycl_diffusion::kernels::Nsd` (check Q5_K) |
| GGUF + safetensors readers, tokenizer | `nextsycl_gguf`, `nextsycl_tok` |
| a 2D VAE decoder | `image/mingimage/src/vae.rs` is Wan's (RMS norm); FLUX.2's is a group-norm SD-style VAE - new |
| gated DeltaNet prefill (conv, L2 norm, delta-rule scan, gates) | `kernels/llm/qwen35` + glm5next's scan: an image engine may not call another kind's library - copy into `kernels/image/mossimage` |
| Qwen3.5 layer layout | `llm/qwen35/src/engine.rs` (read for the tensor roles), the 27B shares the architecture |

### Plan (each step checked against the reference before the next)

0. Reference (`reference/mossimage/ref.py`, CPU float32, venv on hardac1): the DiT's classes from the upstream file
   with the GGUF dequantised into them, the text encoder from the original bf16 shards, the VAE from diffusers.
   Dumps: ids, te_final, sigmas, lat_0, per-step (lat_i, vel_i; vel with the skip), per-block outputs at step 0,
   lat_final, the picture.
1. Engine crate `image/mossimage`, kernels `kernels/image/mossimage`: loader + role check; the DiT; check
   `nextsycl image check` vs the dump (block by block, then step by step).
2. VAE (FLUX.2) and the sampler: latents -> picture.
3. Text encoder (GDN + gated attention) and the tokenizer/template; check vs `te_final`.
4. Registry, model cards (one per quant), `image serve`, the front door (it is `image.generate`; layout JSON is
   just a prompt), docs, README speed table.
5. Silo tuning only after 1-4 are exact.
