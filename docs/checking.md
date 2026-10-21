# Checking it

[← nextsycl](../README.md)

- `nextsycl check <model> <dump dir>`: the forward pass against a llama.cpp dump (`reference/llama-dump`), every
  step compared by cosine, the next token and the top logits.
- `nextsycl batch-check <model> [--prompts "a|b"] [--n N]`: conversations decoded together against each decoded
  alone - the same greedy tokens, the logits exactly equal - and the speed of both.
- `nextsycl spec-check <model> --prompt-file F [--rows 2..5]`: verify passes against one-token decode - must stay
  0 difference (greedy output with MTP equals greedy output without); `--layers [--at N] [--row k]` compares every
  named step of one row against a one-token pass, to find where they part. Check a long prompt too (past 2,048
  tokens the indexer runs: the 3-row drift it found lived there).
- `nextsycl llm bench --needle [--sizes 32768,131072,250000] [--depths 10,50,90]`: a passphrase placed at each depth of
  a long document, asked for at the end - can the running server find it (its context must hold the size).
- `NS_PROFILE=1 nextsycl llm generate ...`: seconds per section, the GPU synced at each boundary; `NS_PROFILE=gpu`: device
  timestamps instead (no syncs - the honest view of decode, where sections are tens of microseconds).
- `NS_PIPE_TRACE=1`: each pipeline stage's time a chunk and the GPUs' free VRAM; `NS_VRAM_GUARD_GIB=G`: a prompt read
  stops with an error before a GPU has less than G free.
- Switches for A/B measurements: `NS_DENSE_F16=0`, `NS_PROMPT_F16_MIN`, `NS_PREFILL_CHUNK`, `NS_HC_FUSED=0`,
  `NS_DECODE_LANES`, `NS_DECODE_DIRECT=1`, `NS_ARENA_MIB`, `NS_PREFETCH`, `NS_LEND=0`, `NS_SPLIT`, `NS_PIPELINE=0`,
  `NS_FUSED_MAX`, `NS_FUSED_DOWN_MAX`, `NS_ESIMD_MAX` (0 = off), `NS_KDA_COLS`, `NS_KV=f16`, `NS_MLA_DEC=0`,
  `NS_SPEC_SAMPLING=0`, `NS_SPEC_DRAFT_TEMP`, `NS_DRAFTS=2`, `NS_NGRAM=K`, `NS_EXPERT_PROFILE`, and
  `NS_FREE_MISSES=1` (wrong output: the speed decode would have with no expert copies).
- `tests/api/`: whole jobs through the front door's API alone, each a worked example and a test of it
  (`kaiju_lighthouse.py`: a painting into a 30 s film with sound, three chained clips joined with ffmpeg).
- `reference/`: the offline studies and microbenchmarks - `expert-cache/` (cache, prefetch, pinning and split
  simulations on decode traces), `expand.cpp`, `esimd-fused.cpp`, `esimd-down.cpp` (the expert kernels alone),
  `mmvq-bw.cpp` (a decode product's bandwidth), `cpu-expert/` (an expert computed on the CPU), `profile/`.
