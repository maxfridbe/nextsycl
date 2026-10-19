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
- **Vision** (`DeepSeek-V4.1-Flash.vision.gguf`, 0.97 GB bf16, the same revision): the ViT and the aligner, the image
  span's three learned rows and the routers' image biases - read from beside the model (`NS_DS41_VISION`), see Vision.
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
| prompt, 16-21 tokens (served, after the first) | ~0.7 s for 41-65 tokens on malformed prompts | 0.6-0.7 s |
| prompt, 2.3K | 144 tok/s | 323-384 tok/s |
| prompt, 8.7K | 203 tok/s | 466-532 tok/s |
| prompt, 35K | 138.5 tok/s | 565-600 tok/s (the needle found) |
| prompt, 318K (1M window) | - | 585 tok/s (479 before the candidate-only indexer), the needle found |
| prompt, 1,025,205 tokens (1M window) | 59 min for 1.07M (302 tok/s, the first served build) | 33 min (520 tok/s), the needle found |
| decode, no drafts (256 tokens; NS_DS41_FREE_GIB=1.5 as served) | 20.5-21.1 tok/s (malformed prompts) | 128K window 21.0-22.9 tok/s, 1M window 20.3-21.3 |
| Engram wait a decode token | 1.3-1.9 ms (the old SATA-class drive) | 0.11 ms |

Run to run the same command varies by about 5-7%; the ranges are of several runs.

What made the prompt pass faster (9cb15f0, da9953f, 4da3366, 2f55e3a, e4f0490):

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
5. **Short prompts' dense products 8 rows a launch** through the decode kernel (below 64 rows the expansion was to
   float32: q_b alone 168 MB a pass).
6. **The indexer's scores on XMX** (deepseek4's kernel copied for 32 heads and the ratio): 35K 476 -> 581 tok/s.
7. **0.75 GiB kept free beyond the plan instead of 1.5** (the plan's pass reserve is generous: 2.2-2.4 GiB measured
   free at the peak): ~100 more experts in VRAM a card, decode +4-9%.

Where a 2K chunk's layer goes (NS_DS41_PIPE_LOG=2): B70 attention 25 ms, its VRAM experts 23-27 ms, the copied waves
51-66 ms (copies and products overlapped); B65 attention 40 ms, VRAM experts 42-51 ms, waves 58-71 ms. A short prompt
moves ~7 GB of host and file experts (the floor of its ~0.6 s).

Tried and left off (each measured; docs/environment.md has the switches): the XMX product with 2 or 4 row tiles and
8 / 16 K slices (within noise end to end), the q4_K nibbles to floats by their bits (B65 one column no faster), a
short prompt's next-layer file reads from its router's guess (no change: 0.6-0.7 s either way), more layers on the
B70 (NS_SPLIT=24 / 28 with the kv source's keys mirrored on the B65: slower - the B70's VRAM is the binding limit),
the VQ decode lookup table's run count. The decode VQ lookup-table variant (each 8 values of x against every
codebook entry in local memory) is timed at load with the others: the B65 takes it (12-bit gate | up 418 -> 328 us).

Round 2 (562da98, e4d273b, 2abd21a):

8. **Decode reads the next layer's likely file-only experts ahead** (its router's guess on this layer's input, chance
   >= 0.15, at most 8, 4 reader threads): the host's turn from router to products had the drive in it - 128K window
   20.4 / 19.9 -> 22.9 / 21.0 tok/s (story / code).
9. **The decode lookup table 1,024 lanes a work-group, 32 runs a row** (B65 12-bit gate | up 328 -> 263 us; the B70
   takes it too for 12-bit gate | up), victims by partial selection, 8 spare VRAM slots a store.
10. **The index sources after the candidate source (24-36) score only their 2,048 candidate blocks** (16K keys) instead of
   every key masked: the same picks (identical text), the B65's 2K chunks flat at 3.3-3.7 s across a 318K prompt
   (3.5 -> 5.1 before); over a 1M prompt the chunks grow 2.5 -> 4.1 s (B70: layers 2, 8, 14, 20 still score every key)
   and 3.7 -> 4.5 s (B65).

Measured and not kept in round 2: decode's file reads after the resident experts' products in the same layer (no
gain: a read outlasts those products), spin-waiting on the router's read back (no gain), the indexer's queries packed in
global memory with 16 key tiles a sub-group (35K 490 vs 599 tok/s; 4 tiles: equal), a grouped one-column q4_K product
for q_b's 20 items a row (B65 93 -> 124 us: slower).

## MTP (the DSpark towers)

Off by default (NS_DS41_MTP=1). Round 2, after the decode gains (256 tokens): code 21.2 without drafts, 20.6 with 1
(82% kept), 20.4 with 2 (78%); prose 22.5 without, 16.8 with 1 (55%), 16.3 with 2 (50%) - still a loss. With well-formed prompts, 256 tokens --ignore-eos, two runs each: code without drafts
19.0 / 18.4 tok/s, 1 draft 18.2 / 18.1 (82% kept), 2 drafts 17.9 / 19.8 (78%); prose 19.9 / 19.0, 1 draft 16.7 / 16.6
(55%), 2 drafts 15.3 / 15.1 (50%). A verify pass's q4_K products now decode each weight once for all its columns
(q4k_cols; spec-check: 0 of 138 three-row rows differ, max |diff| 0), and with the towers loaded the VQ lookup table is
left out (it is built a column at a time). The cost left is the experts: 2-3 tokens route to about twice the experts
of one.

## Placement and memory (served at 131K context)

- VRAM: B70 layers 0-19 (16.8 + 6.7 GiB of experts), B65 layers 20-39 + the head (23.6 GiB); 0.75 GiB kept free
  beyond the plan (2.2 GiB measured free at the peak); the staging ring 0.16 GiB a card.
- Pinned host memory: ~44 GiB of experts + the file slots (0.44 GiB a card).
- The file: ~7 GiB of experts read when needed; the Engram's 203 GB read a row at a time (238K reads/s at QD 48 on the
  P310, 80 us a read alone).

## Vision

Pictures go in through the chat API: OpenAI's `image_url` parts (data URLs), Anthropic's base64 `image` parts, or a
message's `images` (base64, /api/chat); `nextsycl llm generate ... --image PATH` from the command line. The server puts
the engine's marker (`<｜deepseek_image｜>`, id 129264) in each picture's place, and the engine turns each marker into the
picture's span, `[start] ([image] x w, [newline]) x h [end]` (image_processor.py), as ids of its own past the
vocabulary (the picture's key and the place in its span): the same picture gives the same ids, so the prompt cache and
a session's reuse hold. A pass that reads a span writes the span's rows over the placeholder's embeddings: the three
learned rows, and the ViT + aligner's rows in reading order. The span's positions route by the layers' image bias
(`exp_probs_b_vl`), take nothing from the Engram (their rows zero; an n-gram that reaches back into a span is the
pad), and are never routed on the GPU (decode's GPU routing is for text).

- Preprocessing as the reference's: the size plan (min_pixels 544^2, at most 1,024 positions a picture), ImageOps.pad
  with PIL's bicubic resize (crates/diffusion's), the pixels rounded to bf16 as the reference casts them.
- The ViT (32 blocks, 1024 wide, 16 heads, the 2D rotation over the patch grid) and the aligner (3 x 3 patches a row,
  two products with GELU) run in float32 on the first GPU, oneMKL for the products (attention a block of query rows at
  a time, the scores within 96 MB), the weights a layer at a time from the sidecar into the prompt arena; the rows are
  kept with the picture (8 pictures held). 1.0-1.3 s a picture (39 x 39 and 36 x 72 patches) on the B70.
- Parity against the reference's own code (vision.py, image_processor.py of the 1017 release, `refvis.py` loading the
  sidecar's weights) in float32: patches identical, ViT output rel err 1.8e-5 / 2.3e-5, the rows 1.1e-5 / 1.2e-5
  (cosine 1.000000; a 512 x 512 apple and a 1000 x 500 sheet of eight renders). The reference in bf16, its own mode,
  is 3.2e-2 from its float32 rows.
- Answers: the apple described (colour gradient, lenticels, stem, studio backdrop); the sheet as "2 distinct objects,
  each presented from four different angles": the goblin warrior and the ornate helmet on its stand. Through the
  server: a 198-position prompt with a picture 2.5 s cold, 1.1 s again (the picture's rows and the prompt cache).

## Open

- Decode (~20 tok/s against V4's 25.6): the B65's q4_K products are ALU-bound (q_b 252 GB/s against the B70's 478;
  its 1,280 columns leave 4 of 16 lanes a second pass), the VQ products ~20 ms a token on both cards, the swaps' down
  copy before each up (a pool of free VRAM slots would take it off the critical path), the host's turn from router
  logits to queued products (~2.6 ms a token).
- Prompt: the XMX expert product decodes each expert once a 32-token tile; one decode for all of an expert's tokens
  (local memory, many token tiles) is the next kernel.
