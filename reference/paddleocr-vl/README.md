# PaddleOCR-VL references

transformers' own implementation of PaddleOCR-VL on the original BF16 weights (PaddlePaddle/PaddleOCR-VL-1.6 rev
c5630abae1d940eafe0697512a0325494b02ab42), float32 on the CPU - run from that repo's directory (the scripts' `d`):

- `ref.py PICTURE PROMPT [MAX_NEW]` - a greedy reading (use_cache on: the repo's config turns it off, which crawls).
- `dump.py PICTURE PROMPT DIR [N]` - the stages `nextsycl image check DIR --model <the GGUF> [--picture PICTURE]`
  compares: pixel_values, grid.txt, tokens.txt, positions.txt, v_embd, v_layer-*, v_out, projected, l_out-*,
  logits_last, gen.txt.

Note transformers resizes the position grid with align_corners=True; the repo's remote code and llama.cpp use
half-pixel centres. The engine follows transformers (checked: v_embd rel 6e-7).
