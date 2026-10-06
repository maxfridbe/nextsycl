# Notes

## Models to try (2026-10-05)

Start: DogContext/GLM-5.3-Flash-Uncensored-Q2-ds4 (IQ2_XXS + Q2_K, imatrix, MTP; 96.5 GB; sha256
1b3a7d5c36d507bbcb7c99995740772114c5d0ec628a68b8263237b883f98e4e). Then test against:

- AliceThirty/GLM-5.3-Flash-UNCENSORED-GGUF `UD-IQ3_XXS` (120.6 GB; a different uncensoring; Unsloth dynamic)
- our own IQ3_XXS / IQ3_S of GCSA-AiLab/GLM-5.3-Flash-Uncensored-RCO-GSQ-GGUF `Q8/` (310 GB) with
  AesSedai/GLM-5.3-Flash-GGUF `imatrix.gguf` (llama-quantize, ~118-125 GB)
- the GCSA GSQ-RCO 3.5-bit already on the box (137 GB; Q2_K/Q3_K/Q4_K)
- AesSedai/GLM-5.3-Flash-GGUF `IQ3_S` (124.7 GB; the base model, not uncensored) as a quality control

Compare: perplexity on one held-out text, a fixed prompt set (greedy), decode / prompt speed.

## The box

The test box: Arc Pro B70 + B65 (32 GB each, Gen5 x8), Ryzen 9 9950X, 61 GB RAM, nvme0 Gen4 (models), nvme1 Gen3.
Build in the `localhost/h3-build` podman image (oneAPI 2026.1, oneDNN, cargo 1.97) or the `debintel` distrobox.

## Reference runtime

llama.cpp de25343 (PR 27773, glm5-next) built with SYCL in `~/src/llama.cpp-glm5` on the box, two local patches:
the free-memory query warns instead of aborting (it throws in debintel), and one SYCL context per GPU
(`dpct/helper.hpp`, with `GGML_SYCL_DEV2DEV_MEMCPY=2`). Launcher `~/src/run-glm5.sh`. Single-device runs crash in
this build (a range check in the loader) - run it on both cards.
