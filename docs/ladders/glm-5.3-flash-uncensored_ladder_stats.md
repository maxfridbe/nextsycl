# glm-5.3-flash-uncensored-256k: context ladder

Prompt processing (pp) and token generation (tg) by prompt length for `glm-5.3-flash-uncensored-256k` (GLM-5.3-Flash
uncensored IQ2 with its MTP drafts, 256K context, both GPUs), measured 2026-10-27. Port notes: [docs/glm5next.md](../glm5next.md).

## The machine

| | |
|---|---|
| GPUs | Intel Arc Pro B70 (layers 0-21) + Arc Pro B65 (layers 22-44), 32 GB each, xe driver |
| host | AMD Ryzen 9 9950X (16 cores / 32 threads), 61 GB RAM, Fedora, in the `h3-build` image |
| model file | Crucial P310 4 TB NVMe, `GLM-5.3-Flash-Uncensored-IQ2-imatrix-MTP-ds4.gguf` (90 GB) |
| code | nextsycl main at `448d208`: the two GPUs' parts load at once (`NS_LOAD_SEQ=1` for one after the other) |
| expert tiers | B70 3,016 in VRAM (19.9 GiB) + 2,456 in pinned host memory (16.2 GiB); B65 2,842 in VRAM (18.7 GiB) + 4,070 in pinned host memory (26.8 GiB); none on the file only |
| settings | `llm generate --ctx 262144 --temp 0 --gpu 0,1`, MTP drafts on, the front door in maintenance |

Each rung is a fresh process (the load, 32-37 s, not counted). Prompts: this repo's documentation (`pp2k`, `pp8k`,
`pp24k`) and the same with a buried secret code and a question at the end (`needle*`).

## Results

| prompt tokens | pp | tg | tokens generated | needle |
|---:|---:|---:|---:|---|
| 13 | 14.7 tok/s (0.9 s) | 16.5 tok/s | 256 (`--ignore-eos`) | - |
| 2,533 | 471 tok/s (5.4 s) | 15.6 tok/s | 256 | - |
| 9,744 | 691 tok/s (14.1 s) | 17.0 tok/s | 256 | - |
| 24,594 | 872 tok/s (28.2 s) | 17.5 tok/s | 256 | - |
| 34,776 | 949 tok/s (36.7 s) | 18.9 tok/s | 256 (`--ignore-eos`) | found |
| 134,126 | 933 tok/s (143.7 s) | 15.2 tok/s | 256 (`--ignore-eos`) | found |
| 248,495 | 828 tok/s (300.1 s) | 18.0 tok/s | 256 (`--ignore-eos`) | found |

One run of each rung. tg is the average over 256 generated tokens. On the needle prompts the model answers with the code
alone (11-37 tokens); those answers were first measured as they came (tg 14.0, 9.0 and 8.1 tok/s at 35K, 134K and 248K)
and are not a decode rate - a dozen tokens include the first token's wait - so those rungs were re-run with
`--ignore-eos` to generate 256 tokens. Past the end token the text is not an answer, and drafts of it may be accepted
more or less often than a real answer's: read those four tg figures as indicative.

## Reading it

- **pp** climbs from 471 tok/s at 2.5K to ~870-950 tok/s from 25K on and holds at 828 at 248K: a 256K prompt reads in 5 minutes.
- **tg** is 15-19 tok/s at every depth from 13 to 248K tokens. The differences between rungs (15.2 at 134K, 18.9 at 35K,
  18.0 at 248K) are within what the prompt and the drafts' acceptance move, so no trend with context is shown.
- The same cards at an 8K context (a shorter KV reserve, so more experts in VRAM) generated at 19-20 tok/s over 512 tokens
  of prose.
- The secret code was found at 35K, 134K and 248K tokens.
- The cards' arenas peaked at 3.3 and 3.8 GiB on the 248K prompt; no request went past them.
