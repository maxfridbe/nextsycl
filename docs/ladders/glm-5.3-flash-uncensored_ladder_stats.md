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
| 13 | 13.1 tok/s (1.0 s) | 17.62 tok/s | 256 max | - |
| 2,533 | 471 tok/s (5.4 s) | 15.64 tok/s | 256 max | - |
| 9,744 | 691 tok/s (14.1 s) | 17.02 tok/s | 256 max | - |
| 24,594 | 872 tok/s (28.2 s) | 17.45 tok/s | 256 max | - |
| 34,776 | 823 tok/s (42.3 s) | 13.98 tok/s | 320 max | found |
| 134,126 | 935 tok/s (143.4 s) | 8.99 tok/s | 320 max | found |
| 248,495 | 830 tok/s (299.5 s) | 8.06 tok/s | 320 max | found |

One run of each rung. tg is the average over the tokens generated (up to 256 or 320), which are the model's own answer
to that prompt: a short answer such as the secret code alone (the 134K and 248K rungs) is a few tokens, so its tg is the
least certain.

## Reading it

- **pp** climbs from 471 tok/s at 2.5K to ~870-935 tok/s from 25K on and holds near 830 at 248K: a 256K prompt reads in 5 minutes.
- **tg** is 15.6-17.6 tok/s to 25K tokens, 14.0 at 35K, 9.0 at 134K and 8.1 at 248K. Attention over a long context, not the
  experts, is what slows it past 35K.
- The same card at an 8K context (a shorter KV reserve, so more experts in VRAM) generated at 19-20 tok/s over 512 tokens
  of its own prose; the 17.6 here at 13 prompt tokens is the 256K context's reserve showing.
- The secret code was found at 35K, 134K and 248K tokens.
- The cards' arenas peaked at 3.3 and 3.8 GiB on the 248K prompt; no request went past them.
