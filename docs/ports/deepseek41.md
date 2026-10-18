# DeepSeek-V4.1-Flash-1017 (VQ GGUF): port notes

nextsycl runs DeepSeek-V4.1-Flash-1017 as the `llm/deepseek41` engine (the file says `deepseek4`, variant
`v41_flash`), with its kernels in its silo `kernels/llm/deepseek41/silo` (`dist/silo/libnextsycl-deepseek41.so`; the
engine needs it - there is no shared kernel for the VQ experts). Card: `model-cards/llm/deepseek-v4.1-flash-vq.yml`.

## Sources

- **GGUF**: https://huggingface.co/Baekpica/DeepSeek-V4.1-Flash-1017-VQ-GGUF (revision f81ee2e9), 113.6 GB in 40 byte
  ranges (`.gguf.part01-of-40` ...; the card lists them under `parts`: `models pull` joins them after checking each).
  YoungAi's recipe (`vq8sh14-q4k-mtpnative`): the routed experts as 8-value vector codes (12-bit indices, 13 in
  layers 0-13; a shared E4M3 codebook a layer; ~1.6 bits a weight), the rest q4_K, a few fp8 32x32 matrices; the three
  DSpark draft towers inside.
- **Engram tables**: DeepSeek's own shards 47 and 48 of the 1017 checkpoint (2 x 101.5 GB, safetensors), in
  `deepseek-engram/` beside the model (NS_DS41_ENGRAM_DIR moves them); read a row (264 bytes) at a time with O_DIRECT.
- **Vision** (`DeepSeek-V4.1-Flash.vision.gguf`): not ported yet.
- **License**: MIT (DeepSeek), YoungAi's tools MIT.

## The file has no token types

The GGUF carries `tokenizer.ggml.tokens` and merges but no `tokenizer.ggml.token_type`. Without it the tokenizer
took no token as special, and the chat template's markers (`<｜begin▁of▁sentence｜>`, `<｜User｜>`,
`<｜Assistant｜>`, `</think>`) were split into ordinary pieces: answers looped, the model wrote `<｜endoftext｜>` as
text and never stopped. `crates/tok` now takes DeepSeek's added tokens by their form (`<｜...｜>`, the think tags)
when a file has no types (c4f5724). Everything measured before that (the earlier decode and MTP numbers) ran on
malformed prompts; the parity checks did not (they read the reference's token ids).

## Parity

- `llm check <model> r16` (16 tokens, the CPU reference `reference/deepseek41`, NS_DS41_QSIM=0): every layer within
  1e-4 without the key simulations (0ac12bd); with the prompt path's XMX expert product (da9953f) the check matches the
  expansion path's: result_output cos 0.99620 (0.99620 before), decode vs prompt 2.0e-4 (2.5e-4), the same next token
  ("Paris").
- 1,100 tokens teacher-forced (the indexer picking): cos >= 0.9993.
- `llm kernels`: the VQ decode variants against the host product 1.4e-7; `vq_gemm` (prompt, 40 tokens, gate | up as one
  matrix and down) 1.2-1.6e-7.
- `spec-check --prompt-decode` (a prompt pass against the same prompt fed a token at a time, 2,268 tokens): cos 0.992,
  the same top token (the expansion path: 0.993).
- Verify rows bit-identical to one-token rows (spec-check, 0 of 46 rows differ: 17242f5).

## Speed (B70 + B65, both Gen5 x8, 61 GB RAM, the files on a Crucial P310 4 TB)

| | before (60e5c08) | now |
|---|---|---|
| prompt, 39 tokens | 12.8 tok/s | 34 tok/s |
| prompt, 2.3K | 144 tok/s | 348 tok/s |
| prompt, 8.7K | 203 tok/s | 463-510 tok/s |
| prompt, 35K | 138.5 tok/s | 476-481 tok/s |
| decode (served, no drafts) | 20.5-21.1 tok/s | 19.3-21.9 tok/s |
| Engram wait a decode token | 1.3-1.9 ms (2TB SATA-class NVMe) | 0.11 ms |

What made the prompt pass faster (9cb15f0, da9953f):

1. **The experts on XMX straight from their indices** (`ns_ds41_vq_gemm`): each lane decodes its weight row's 16 values
   a step into a packed fp16 B tile's registers (the layout glm5next probed), 4 K slices a work-group, 8 token tiles a
   sub-group past 32 tokens. The expansion to fp16 (a 70 MB write and read an expert, whatever its tokens) is gone.
2. **A VRAM staging ring** (24 slots): the experts in pinned host memory are copied up on the copy lane, a wave at a
   time, while the VRAM-resident ones are multiplied; before, kernels read host memory in place (~12 GB/s) and the
   file's experts were swapped in (evicting others down).
3. **Expert-major placement at load**: in layer order the last layers of each GPU lived in the file alone (300-460 ms
   a layer of a 2K chunk); now every layer has the same share of VRAM, pinned memory and the file.
4. **The file's experts read on 8 threads** into 64 pinned slots while the GPU works; each chunk's Engram rows asked
   for a chunk ahead.

Where a 2K chunk's layer goes now (NS_DS41_PIPE_LOG=2): attention 25 ms (B70) / 40 ms (B65), experts 80-100 / 95-135
ms; the B65 bounds the pipeline. The decode products got a lookup-table variant (each 8 values of x against every
codebook entry in local memory), timed at load with the others: the B65 takes it (12-bit gate | up 418 -> 328 us).

## MTP (the DSpark towers)

Off by default (NS_DS41_MTP=1 loads them): measured before the tokenizer fix, drafts 2 kept 80% on code but ran at
16.5 tok/s against 18.0 without; 43% on prose (12.6 against 18.7). A verify pass of 3 rows costs ~2.7 one-token passes:
3 tokens route to ~2.5x the experts, and the q4_K decode products re-decode their weights for each column. To
re-measure with well-formed prompts.

## Placement and memory (served at 131K context)

- VRAM: B70 layers 0-19 (17.1 + 6.8 GiB of experts), B65 layers 20-39 + the head (23.9 GiB); 1.5 GiB kept free a
  card; the staging ring 0.16 GiB a card.
- Pinned host memory: 40 GiB of experts (NS_DS41_PIN_GIB) + the file slots (0.44 GiB a card).
- The file: ~10 GiB of experts read when needed; the Engram's 203 GB read a row at a time (238K reads/s at QD 48 on the
  P310, 80 us a read alone).

## Open

- Prompt: the per-layer copies and products still add up rather than overlap fully; a chunk-wide expert prefetch or
  a split putting more layers on the B70 (the kv-source layers 2, 8, 14, 20 bound where a part may begin) are next.
- MTP: a q4_K decode product that decodes each weight once for all of a verify pass's columns.
- Vision (the sidecar GGUF).
