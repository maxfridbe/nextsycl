# nemotron-3.5-super-iq1m: context ladder

Prompt processing (pp) and token generation (tg) by prompt length for `nemotron-3.5-super-iq1m` (Nemotron 3.5 Super
120B-A12B, Unsloth UD-IQ1_M, both GPUs, MTP drafts), measured 2026-11-05 on hardac1.

## The machine

| | |
|---|---|
| GPUs | Intel Arc Pro B70 (layers 0-61) + Arc Pro B65 (layers 62-87), 32 GB each, xe driver; 42.7 GiB of weights over both |
| host | AMD Ryzen 9 9950X (16 cores / 32 threads), 61 GB RAM, Fedora, in the `h3-build` image |
| model files | Crucial P310 4 TB NVMe, `NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-UD-IQ1_M-0000{1,2}-of-00002.gguf` |
| silo | `dist/silo/libnextsycl-nemotronh.so` (the MoE layers) |
| code | nextsycl main at `73ed46a`, the production build (`~/src/nextsycl/dist/nextsycl`) |
| settings | `llm generate --ctx 32768 --temp 0 --gpu 0,1`, MTP drafts on (2 a step), prompt chunks of 2048; the front door in maintenance, the GPU lock held |

Each rung is a fresh process (the load, 24-58 s, not counted). Prompts: this repo's documentation (`pp2k`, `pp8k`,
`pp24k`) and the same with a buried secret code and a question at the end (`needle25k`), without any template text in
them: `llm generate` puts each model's own chat template around them. 32,768 is the registry card's context, so the
ladder stops at 27K.

## Results

| prompt tokens | pp | tg | tokens generated | drafts accepted | needle |
|---:|---:|---:|---:|---:|---|
| 25 | 58 tok/s (0.4 s) | 45.8 tok/s | 39 (its answer) | 26 of 27 | - |
| 2,705 | 1,014 tok/s (2.7 s) | 42.5 tok/s | 256 | 156 of 185 | - |
| 10,337 | 1,415 tok/s (7.3 s) | 43.5 tok/s | 256 | 161 of 185 | - |
| 26,264 | 1,548 tok/s (17.0 s) | 40.4 tok/s | 256 | 156 of 188 | - |
| 27,408 | 1,542 tok/s (17.8 s) | 44.6 tok/s | 85 (its answer) | 57 of 60 | found |
| 27,408 | 1,549 tok/s (17.7 s) | 46.3 tok/s | 256 (`--ignore-eos`) | 167 of 174 | found |

One run of each rung; tg is the average over the tokens generated.

## Reading it

- **pp** climbs from ~1,000 tok/s at 2.7K to ~1,550 tok/s from 26K on: 27K tokens read in 18 s.
- **tg** holds at 40-46 tok/s at every depth to 27K: 40 of the 88 layers are Mamba-2 (no cache to read) and only 8 are
  attention, so context barely costs a token anything. The drafts' acceptance (83-96%) moves it more than depth.
- The secret code was found at 27K, with a short think first.
- Not run: past 32K (the card's context; the README's 128K figure, pp 1,120 / tg 38, is from the port's own test).
