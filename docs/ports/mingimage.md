# Ming-Image 0.1 Design (text to image for designs)

`image/mingimage`, from nextcuda (its 2026-10-19 port, `5e710bc`): inclusionAI's Ming-Image-0.1-Design - Ling-Mini-2.0
(BailingMoeV2, 256 experts routed on the host), the bidirectional Qwen2 connector and its projections; Z-Image's 6B DiT
(refiners, 30 blocks, sandwich norms, the direct condition after the query features); Wan 2.1's RGBA decoder. Every
weight stays on the card in its GGUF's type. The Rust is nextcuda's; its CUDA kernels (`mi.cu`) are SYCL here
(`kernels/image/mingimage/mi.cpp`, the same names and arithmetic) plus a strided copy, and nextcuda's shared
`nsd_linear_q` (a GGUF matrix multiplied a slice of rows at a time) is composed in the engine from nextsycl's shared
dequant and product (`ffi::linear_q`) - no shared kernel changed.

On nextsycl's 32 GB cards everything is on one card (nextcuda splits the text encoder onto a second 12 GB card;
`--gpu 0,1` still does that here): 19.4 GiB at the peak of a 2048 x 2048 picture on a B70.

Files: realrebelai's GGUFs (DiT Q8_0, text encoder Q4_K_M) and Comfy-Org's VAE, `model-cards/image/ming-image-0.1-design.yml`
(17.5 GB). Use: `image.generate` with `model: ming-image-0.1-design`, `nextsycl image gen ... --model ming-image-0.1-design`.

## Checked (2026-10-19, B70)

Against `reference/mingimage/ref.py` (inclusionAI's DiT class at 3515cd7b and its text side, float32 on the CPU, the
same files; ComfyUI's Wan VAE) at 512 x 512, 12 steps, seed 7 - `nextsycl image check <dump> --model ming-image-0.1-design`:

| stage | here | nextcuda (RTX 5070) |
|-------|------|---------------------|
| tokens | identical (314) | identical |
| direct condition | 9.0e-3 | 8e-3 |
| query features (connector) | 4.5e-2 (routing ties: up to 63 of 256 x 256 choices differ by layer 19) | 5e-2 |
| context / noise refiner, layer 0 | 1.9e-4 / 2.2e-4 | - |
| velocity, steps 0 and 5 | 2.2e-3 / 1.3e-3 | 1e-3 |
| latents after 12 steps | 1.7e-2 (cos 0.99986) | picture 1.5e-2 |
| VAE (RGBA) | 1.6e-4 | 2.4e-4 |

Speed (B70, one card, 12 steps): 1024 x 1024 in 9.9 s (0.81 s a step), 2048 x 2048 in 43.8 s (3.8 s a step); nextcuda
17.7 s and 164 s on the RTX 5070 + 3060. Load 4.5 s from the page cache.
