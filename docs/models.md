# Supported models

[← nextsycl](../README.md)

What `nextsycl models search` lists and `nextsycl models pull <id>` installs: a model card each
(`model-cards/<kind>/<id>.yml` - direct links, sizes and SHA-256, the registry settings, the prompt guide; read at
run time, not built in: [model-cards/README.md](../model-cards/README.md)). **Checked**: compared stage by stage with a reference run on the same files. **Served**:
running in daily use here, with its speed measured. **Listed**: in the catalog, not yet run on this machine.

**Language** (`nextsycl llm`)

| id | model | engine | size | status |
|---|---|---|---|---|
| `glm-5.3-flash-uncensored`, `-128k`, `-256k` | GLM-5.3-Flash uncensored IQ2, 64K / 128K / 256K context, tool calls (its `<tool_call>` form) | `glm5next` | 89.9 GiB | checked (llama.cpp), served on both cards |
| `deepseek-r1-iq1s` | DeepSeek-R1 671B UD-IQ1_S, 32K: experts in VRAM, pinned RAM (on the CPU) and the NVMe | `deepseek2` | 130.6 GiB | checked (llama.cpp), served on both cards |
| `deepseek-v4-flash-iq2m`, `-256k` | DeepSeek-V4-Flash 284B UD-IQ2_M, 128K / 256K + 32K: sliding window + compressed attention (CSA / HCA, the lightning indexer), mHC, experts in VRAM and pinned RAM, MTP drafts (the original checkpoint's draft block, converted after the pull); a needle found at 35K, 140K, 261K and 1,071,766 tokens | `deepseek4` | 88.3 GiB | checked (llama.cpp), served on both cards |
| `deepseek-v4.1-flash-iq2xxs`, `-fast` | DeepSeek-V4.1-Flash-1017 (Unsloth UD-IQ2_XXS), 128K: V4's attention with the indexer at ratio 1/2 and shared compressed keys, mHC, the Engram (n-gram rows from the table inside the file), experts in IQ1_S-Q2_K by layer on the std kernels (bit-exact against ggml), pictures through the vision sidecar; `-fast`: cold low-weight experts left out and attention in q4_K (tg ~25 against ~14 tok/s sustained, perplexity +3-5%); [docs/ports/deepseek41-unsloth.md](ports/deepseek41-unsloth.md) | `deepseek41` | 146 GiB of experts + 126 GiB Engram | checked (ggml's dequantize_row_*), served on both cards |
| `deepseek-v4.1-flash-q2kxl`, `-fast`; `deepseek-v4.1-flash-iq1m`, `-fast`; `deepseek-v4.1-flash-q3kxl` | other quants of the same model (secondary; [docs/ports/deepseek41-unsloth.md](ports/deepseek41-unsloth.md#other-quants)): UD-Q2_K_XL steadier on long answers, tg 8.8 exact / 19.2 `-fast`; UD-IQ1_M experimental (perplexity +9%, repeats from ~5,000 words, as in llama.cpp), tg 16.2 / 25.8; UD-Q3_K_XL near the original, tg ~4.4 | `deepseek41` | 178 / ~122 / 253 GiB of experts + the Engram (126 GiB Q5_K / 103 GiB IQ4_NL / 189 GB MXFP8) | checked (IQ3_S split-Q8 against float32); served on both cards |
| `deepseek-v4.1-flash-xq`, `-fast` | DeepSeek-V4.1-Flash-1017 XQ v3, uncensored (IndrasMirror; routed experts each Q3_K / Q2_0 / PTQ1_0, 2.09 bits a weight), 128K, tg 8.2 exact / 15.7 `-fast`; [docs/ports/deepseek41-xq.md](ports/deepseek41-xq.md) | `deepseek41` | 142 GiB of experts + DeepSeek's Engram (2 x 101.5 GB) | checked (each type against the host); served on both cards |
| `deepseek-v4.1-flash-vq` | the earlier VQ GGUF (1.6-bit experts; repeated itself past ~2K tokens): file removed, card kept; [docs/ports/deepseek41.md](ports/deepseek41.md) | `deepseek41` | 105.8 GiB + 203 GB Engram | retired |
| `nemotron-3.5-super-q2kxl` | NVIDIA Nemotron 3.5 Super 120B-A12B UD-Q2_K_XL (40 Mamba-2, 40 latent-MoE, 8 attention layers; 512 experts, 22 a token), 32K, MTP drafts; text only (the vision tower not yet) | `nemotronh` | 56 GiB | checked (llama.cpp), served |
| `nemotron-3.5-super-iq1m` | the same, UD-IQ1_M, on both GPUs | `nemotronh` | 43 GiB | checked (llama.cpp), served |
| `nemotron-3.5-super-iq1m-b70` | the same on the B70 alone, the experts that do not fit in pinned host memory (the B65 left free) | `nemotronh` | 43 GiB | served |
| `qwen3.8-flash-next-iq2_xs` | Qwen3.8-Flash-Next IQ2_XS | `qwen4exp` | 63.4 GiB | served (verify passes and batches exact) |
| `qwen3.8-flash-next-iq2_xs-uncensored` | the same with the refusal projection (a control vector) | `qwen4exp` | 63.4 GiB | served |
| `qwen3.8-flash-next-coder-iq1_m`, `-uncensored` | Qwen3.8-Flash-Next Coder IQ1_M | `qwen4exp` | 54.4 GiB | served |
| `swift-1.5-iq2_xs` | Swift 1.5 IQ2_XS (short thinking) | `qwen4exp` | 63.5 GiB | served |
| `qwen3.8-27b-q8` | Qwen3.8-27B Q8_0 (dense: gated DeltaNet + gated full attention), 32K, MTP drafts, pictures (its mmproj) | `qwen35` | 28.4 GiB | checked (llama.cpp), served (verify passes exact) |
| `qwen3.8-27b-q4` | Qwen3.8-27B UD-Q4_K_M, 128K + 32K, MTP drafts, pictures (its mmproj) | `qwen35` | 16.6 GiB | checked (llama.cpp), served (verify passes exact) |
| `qwen3.8-27b-q3` | Qwen3.8-27B UD-Q3_K_XL, 128K + 32K, MTP drafts | `qwen35` | 13.5 GiB | served |
| `qwen3.8-27b-q8-256k` | Qwen3.8-27B Q8_0 on both GPUs, 256K + 32K, MTP drafts | `qwen35` | 28.4 GiB | served (verify passes exact) |
| `qwen3.8-27b-q4-256k` | Qwen3.8-27B UD-Q4_K_M on both GPUs, 256K + 32K, MTP drafts | `qwen35` | 16.6 GiB | served (verify passes exact) |
| `muse-glimmer-30b-q8` | Muse Glimmer 30B Abliterated Q8_0 (dense gated attention: 39 sliding-window layers of 2,048, 13 full; post norms, soft-capped logits), 128K + 32K; its DFlash drafter (15 drafts a step, verify rows exact: the same text as without); its reasoning as `reasoning_content` | `muse-glimmer` | 29.1 GiB | checked (llama.cpp, 6K tokens), served |
| `muse-glimmer-30b-q8-b70`, `muse-glimmer-30b-q8-b65` | the same on the B70 or the B65 alone, no drafter (the engine drops it anyway when it does not fit), 128K + 32K | `muse-glimmer` | 26.2 GiB | served |
| `muse-glimmer-30b-q4`, `muse-glimmer-30b-q6` | Muse Glimmer 30B Q4_K_M / Q6_K on the B70 alone with its DFlash drafter (the silo's K-quant products on XMX, verify exact), 128K + 32K | `muse-glimmer` | 15.8 / 21.3 GiB + 1.5 | checked (spec-check 0), served |

**3D** (`nextsycl model3d`: a mesh from a picture)

| id | model | engine | size | status |
|---|---|---|---|---|
| `hunyuan3d-2mini-turbo` | Hunyuan3D-2mini-turbo: the 0.6B Flux-style DiT, DINOv2-giant, the FlashVDM turbo decoder; shape only | `hunyuan3d` | 3.9 GiB | checked |
| `hunyuan3d-2.1` | Hunyuan3D 2.1: the 3.0B MoE DiT (shape) and the paint model (PBR textures) | `hunyuan3d` | 15.0 GiB | checked |

**Image** (`nextsycl image`; sizes with the text encoder, VAE and tokenizer)

| id | model | engine | size | status |
|---|---|---|---|---|
| `qwen-image-2.1-q8` | Qwen-Image 2.1 Q8_0: text to image (RGBA), edits and compositions of up to 8 pictures | `qwenimage21` | 16.4 GiB | checked, served |
| `hunyuanimage-2.1-distilled-q8` | HunyuanImage 2.1 Distilled (MeanFlow, guidance embedded): text to image at 2K, 8 steps | `hunyuanimage-2.1` | 24.4 GiB (25.3 GiB at 2048x2048) | checked |
| `hunyuanimage-2.1-q8` | HunyuanImage 2.1 Q8_0: 50 steps, a negative prompt's guidance (APG) | `hunyuanimage-2.1` | 24.3 GiB | checked |
| `unlimited-ocr-q8`, `unlimited-ocr-q4` | Unlimited-OCR (Baidu, the DeepSeek-OCR architecture): pages to Markdown, tables as HTML, boxes with `<|grounding|>`; long pages in fixed memory | `unlimited-ocr` | 6.5 GiB (decoder in half) | checked (PIL, deepencoder.py, llama.cpp), served |
| `paddleocr-vl-1.6` | PaddleOCR-VL 1.6 (PaddlePaddle): SigLIP NaViT at the page's size + ERNIE-4.5 0.3B; text, tables, formulas, charts, seals, spotting | `paddleocr-vl` | ~3 GiB (vision float32, decoder half) | checked (transformers), served |
| `olmocr-2-7b-q8` | olmOCR-2-7B-1025 (Ai2, a Qwen2.5-VL-7B fine-tune): pages to Markdown in reading order | `olmocr2` | ~11 GiB (decoder Q8_0, vision float32) | checked (llama.cpp), served |
| `qwen-image-2.1-turbo-q8` | Qwen-Image 2.1 Turbo (Viggle v0.3 merged): 6 steps | `qwenimage21` | 16.5 GiB | checked |
| `qwen-image-2.1-q6k`, `-uncensored-q8` | Qwen-Image 2.1 Q6_K; the uncensored model in Q8_0 | `qwenimage21` | 14.8 / 16.4 GiB | listed |
| `qwen-image-2.1-q4km`, `-uncensored-q4km` | Q4_K_M | `qwenimage21` | 13.6 GiB | listed (needs Q5_K in the dequant) |

**LoRAs** for Qwen-Image 2.1 (`--lora NAME[:SCALE]`, merged at load)

| id | what | status |
|---|---|---|
| `qwen-image-2.1-uncensored-lora` | the uncensored change, rank 16 | checked |
| `qwen-image-2.1-turbo-lora` | Viggle's 6-step turbo, rank 128 (brings its sigmas) | served |
| `qwen-image-2.1-pruna-8step-lora` | Pruna's 8-step distill (brings its sigmas) | checked |
| `qwen-image-2.1-pruna-5step-lora` | Pruna's 5-step distill | listed |

**Video** (`nextsycl video`: text, pictures, a voice and other clips to video with sound)

| id | model | engine | size | status |
|---|---|---|---|---|
| `minimax-h3` | MiniMax H3: the 50-block FL2VA DiT in int8 ConvRot (the fastest), the Qwen3-VL 32B text encoder (Q4_K_M), video and audio VAEs, the latent upscaler | `video/h3` | 42.6 GiB | served (H3's own engine moved in; checked against it) |
| `minimax-h3-q6k`, `-q4km` | the denoiser as a Q6_K / Q4_K_M GGUF: less of the card, longer clips | `video/h3` | 38.5 / 33.6 GiB | served by H3 |
| `minimax-h3-q8` | the denoiser as a Q8_0 GGUF (a reference form) | `video/h3` | 43.0 GiB | listed |
| `minimax-h3-realism-lora`, `-ref2v-turbo-lora`, `-lms-lora` | LoRAs: realism (people), ref2v in 4 steps, LMS | | | listed (realism: served by H3) |

**Audio** (`nextsycl audio`: songs from lyrics and a description; speech from text)

| id | model | engine | size | status |
|---|---|---|---|---|
| `minimax-music3` | MiniMax Music 3: the 8B semantic language model and RVQ depth decoder (half on the card), the 2.4B flow transformer, the Flow-VAE decoder; up to 6 minutes, 44.1 kHz stereo | `minimaxmusic3` | 26.6 GiB | checked, served |
| `minimax-music3-int8` | the same files, the language model and depth decoder in int8 at load: half the VRAM, 1.7x the frames a second | `minimaxmusic3` | 26.6 GiB | checked, served |
| `qwen3-tts-custom` | Qwen3-TTS 12 Hz 1.7B CustomVoice: nine built-in voices, styled by an instruction; 10 languages, 24 kHz | `qwen3tts` | 4.5 GiB | checked, served |
| `qwen3-tts-design` | Qwen3-TTS VoiceDesign: a voice made up from a description | `qwen3tts` | 4.5 GiB | checked, served |
| `qwen3-tts-base` | Qwen3-TTS Base: a voice cloned from a recording (its x-vector; with its transcript, in context) | `qwen3tts` | 4.5 GiB | checked, served |
| `voxcpm2` | VoxCPM2 2B: speech in 30 languages at 48 kHz - plain, designed from a description, cloned (with a transcript: continued); MiniCPM4 language models, a local flow-matching DiT, AudioVAE V2 | `voxcpm2` | 4.6 GiB | checked, served |
| `voxcpm2-int8` | the same files, the language models in int8 at load | `voxcpm2` | 4.6 GiB | checked, served |
| `breeze-tts-2` | Breeze TTS 2 (~3B, Q8_0 GGUF): English and Chinese at 24 kHz - voices designed, cloned (recording + transcript) or directed (a clone + an instruction), vocal events; weights and outputs non-commercial (BreezeBlue license) | `breezetts2` | 3.4 GB + codec | checked |
| `vibevoice-1.5b` | VibeVoice-1.5B (Q8_0 GGUF): long-form speech of up to four speakers in English and Chinese at 24 kHz, each voice cloned from a recording (no transcript); MIT, with Microsoft's responsible-use terms | `vibevoice` | 3.9 GiB | checked |
| `breeze-tts-2-q4` | the same model from the Q4_K GGUF (a 2.2 GB download; the same memory on the card; the 4-bit text encoder is 1e-1 from upstream - a word can slip) | `breezetts2` | 2.2 GB + codec | checked |
| `s2-pro` | Fish Audio S2 Pro (~4.6B Dual-AR, Q8_0 GGUF with its codec): ~80 languages at 44.1 kHz - voices cloned (recording + transcript), free-form inline tags (`[whisper]`, `[laughing]` ...; `instructions` become one), several speakers in one text; no voice design; weights and outputs non-commercial (Fish Audio Research License) | `s2pro` | 5.6 GB | checked |
| `voxtral-4b-tts` | Voxtral 4B TTS 2603 (Mistral, Q8_0 GGUF): nine languages at 24 kHz in 20 preset voices; no cloning from a recording, design or instructions (the open weights have no audio encoder); saved voices enrolled for it (experimental); weights, voices and outputs non-commercial (CC BY-NC 4.0) | `voxtraltts` | 4.3 GB | checked |

**3D** (`nextsycl model3d`: rigging and animation of a mesh; the result keeps the input mesh, its UVs and textures)

| id | model | engine | size | status |
|---|---|---|---|---|
| `animateanymesh` | AnimateAnyMesh: a mesh set in motion by a sentence - DyMeshVAE (neighbour attention over the mesh's edges), a rectified-flow MMDiT on SD 1.5's CLIP-L; 16 frames as glTF morph targets (Apache-2.0) | `animateanymesh` | 0.9 GiB | checked |
| `unirig` | UniRig: a skeleton (an OPT-350m-shaped decoder over a Michelangelo shape encoding, 15 sampled beams) and skin weights (Point Transformer V3, bone attention, a voxel geodesic prior) for any mesh (MIT) | `unirig` | 5.4 GiB | checked |
| `rignet` | RigNet: a skeleton (GMEdgeNet joints, RootNet, BoneNet's tree) and skin weights (SkinNet over volumetric geodesics) for a character mesh; **GPLv3**, its own crate | `rignet` | 0.2 GiB | checked |
**3D** (`nextsycl model3d`: a picture to a textured mesh, GLB)

| id | model | engine | size | status |
|---|---|---|---|---|
| `trellis2` | TRELLIS.2 4B Q8_0: DINOv3, four 1.3B flows (structure, shape 512 + 1024, texture 1024), the sparse decoders; the 1024 cascade, PBR textures; BiRefNet cuts opaque pictures out | `trellis2` | 8.8 GiB (9.1 GiB peak on the card) | checked |
| `trellis2-512` | the same, the 512 path only (the 512 shape and texture flows) | `trellis2` | 7.4 GiB (2.9 GiB peak) | checked |
| `pixal3d` | Pixal3D Q8_0: TRELLIS.2 with pixel-aligned DINOv3 features (manual field of view; NAF not ported) | `pixal3d` | 8.9 GiB (4.5 GiB peak) | checked |
