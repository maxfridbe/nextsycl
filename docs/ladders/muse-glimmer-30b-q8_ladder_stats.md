# muse-glimmer-30b-q8: context ladder

Prompt processing (pp) and token generation (tg) by prompt length for `muse-glimmer-30b-q8` (Muse Glimmer 30B
abliterated, dense Q8_0, both GPUs, its DFlash drafter), measured 2026-11-05 on hardac1.

## The machine

| | |
|---|---|
| GPUs | Intel Arc Pro B70 (the drafter and the first layers) + Arc Pro B65, 32 GB each, xe driver; 27.7 GiB of weights over both |
| host | AMD Ryzen 9 9950X (16 cores / 32 threads), 61 GB RAM, Fedora, in the `h3-build` image |
| model files | Crucial P310 4 TB NVMe, `Muse-Glimmer-30B-Abliterated-Q8_0.gguf`, the drafter `dflash-Muse-Glimmer-30B-Abliterated-Q4_K_M.gguf` beside it |
| silo | `dist/silo/libnextsycl-museglimmer.so` (Q8_0 repacked), `NS_SILO_NAME=museglimmer` as the registry card sets |
| code | nextsycl main at `73ed46a`, the production build (`~/src/nextsycl/dist/nextsycl`) |
| settings | `llm generate --ctx 131072 --temp 0 --gpu 0,1`, DFlash drafts on (15 a step; 5 layers reading the target's layers 2, 14, 26, 38, 50), prompt chunks of 2048; the front door in maintenance, the GPU lock held |

Each rung is a fresh process (the load, 13-45 s, not counted). Prompts: this repo's documentation (`pp2k`, `pp8k`,
`pp24k`) and the same with a buried secret code and a question at the end (`needle35k`, `needle100k`), without any
template text in them: `llm generate` puts the model's own chat template around them (a "Hi" is 541 tokens with it).

## Results

| prompt tokens | pp | tg | tokens generated | drafts accepted | needle |
|---:|---:|---:|---:|---:|---|
| 541 | 697 tok/s (0.8 s) | 40.8 tok/s | 93 (its answer) | 64 of 92 | - |
| 2,999 | 962 tok/s (3.1 s) | 33.4 tok/s | 256 | 164 of 255 | - |
| 10,078 | 962 tok/s (10.5 s) | 29.9 tok/s | 256 | 160 of 255 | - |
| 24,679 | 837 tok/s (29.5 s) | 28.5 tok/s | 256 | 161 of 255 | - |
| 34,656 | 792 tok/s (43.7 s) | 50.4 tok/s | 77 (its answer) | 62 of 76 | found |
| 34,656 | 797 tok/s (43.5 s) | 64.5 tok/s | 256 (`--ignore-eos`) | 218 of 257 | found |
| 99,587 | 526 tok/s (189.2 s) | 41.2 tok/s | 71 (its answer) | 58 of 71 | found |
| 99,587 | 532 tok/s (187.3 s) | 54.5 tok/s | 256 (`--ignore-eos`) | 225 of 261 | found |

One run of each rung; tg is the average over the tokens generated.

## Reading it

- **pp** is ~960 tok/s to 10K, then falls with depth as attention grows: 837 at 25K, ~795 at 35K, ~530 at 100K
  (a 100K prompt reads in a bit over 3 minutes). This is below the README's ~1,350 (1,215 at 35K) for the same card;
  that figure is from the port's own runs and was not re-measured here, so the gap is not explained yet.
- **tg** follows the drafts more than depth: 28-33 tok/s on the open-ended pp rungs (~63% of drafts accepted) and
  54-64 tok/s on the needle rungs (84-87%), where the answer and the text past it repeat the prompt's words. Read the
  `--ignore-eos` figures as indicative: past the end token the text is not an answer.
- The secret code was found at 35K and 100K, after a short think on its to=self channel.
- Not run: 128K (the card's 131,072 leaves no room for a 128K prompt and an answer).
