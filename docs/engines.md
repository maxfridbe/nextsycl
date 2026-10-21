# The engines

[← nextsycl](../README.md)

A Rust runtime for language, image, video and audio models on Intel Arc GPUs, with SYCL kernels. Each kind of model has
a contract its server and command line drive, and each model architecture its own engine behind it, tuned end to end -
a library another program can use without the server (`docs/architecture.md`). New models are welcome:
`CONTRIBUTING.md` walks through a port, from each kind's template engine. The language models today: **GLM-5.3-Flash**
(`glm5-next`, `llm/glm5next`): 45 layers of KDA and MLA attention with a DSA lightning indexer, 288 experts, and
hyper-connections. The experts that do not fit the cards live in pinned host memory, and the GPU reads them over PCIe.
It runs on one card or splits the layers over several. And **Qwen3.8-Flash-Next** (`qwen4exp`, `llm/qwen4exp`): Gated
DeltaNet and QSA attention, hyper-connections, a hashed per-layer embedding and 512 experts, on the Strata SYCL port's
kernels - every weight in VRAM over the two cards or on one with its cold experts in pinned host memory, the MTP draft
layer, the window and the drafter as SYCL graphs; verify passes and batches bit-exact, greedy output with drafts equal
to output without. Every model's speed at each context length: [Speed by model](speed.md#speed-by-model). **Qwen-Image 2.1**
(`image/qwenimage21`): text to image, its Qwen3-VL text encoder, 7B DiT and RGBA VAE all in SYCL, checked stage by
stage against the reference on the same quantized files; 1024x1024 in 40 steps takes 23.8 s on the B70 (16.5 s with
int8 weights), 37.5 s on the B65. **MiniMax H3** (`video/h3`): text, pictures, a voice and other clips to video with
sound - H3's SYCL engine moved in whole (its jobs, daemon, studio and tools as `nextsycl video ...`), with its own
speed: a 896x672, 4.5 s clip in 87-95 s on the B70 (8 s a step), H3's 98 s. **MiniMax Music 3**
(`audio/minimaxmusic3`): songs from lyrics and a description, up to six minutes of 44.1 kHz stereo - its 8B semantic
language model, RVQ depth decoder, 2.4B flow-matching transformer and Flow-VAE decoder in SYCL, checked stage by stage
against MiniMax's diffusers code on the same files; a minute of song in 89 s on the B70 with the language model in int8
(118 s in half). **Qwen3-TTS** (`audio/qwen3tts`): speech in ten languages - nine built-in voices styled by an
instruction, voices made up from a description, voices cloned from a recording - its talker, code predictor, 12 Hz
codec (decoder and encoder) and speaker encoder in SYCL, checked against qwen-tts' own code; a minute of speech in 10.7
s on the B70 (8.2 s in int8). **Breeze TTS 2** (`audio/breezetts2`): English and Chinese speech - voices designed from
a description, cloned from a recording and its transcript, or cloned and directed by an instruction, vocal events
inline ("(laugh)", "[笑]") - its T5Gemma2 text encoder, Qwen3 backbone and depth decoder from CrispASR's GGUF on the
music engine's kernels and its own, the 12 Hz codec shared with Qwen3-TTS; checked against upstream's code; 1.8-2.2x
real time on the B70. **VibeVoice** (`audio/vibevoice`): long-form speech of up to four speakers, each voice cloned
from a recording without its transcript - its Qwen2.5-1.5B language model, diffusion head and streaming σ-VAE decoder
and semantic encoder in SYCL from gguf-org's GGUF, checked stage by stage against transformers' implementation; 10.6x
real time on the B70. **Voxtral TTS** (`audio/voxtraltts`): Mistral's Voxtral 4B TTS - nine languages in 20 preset
voices - its Ministral 3B backbone, flow-matching acoustic transformer and Voxtral codec decoder from CrispASR's GGUF,
checked against vllm-omni's code; 2.3x real time on the B70 (the open weights have no audio encoder: no cloning from a
recording; saved voices enrolled by fitting their codes on the GPU, experimental).
