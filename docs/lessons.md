# Lessons learned

[← nextsycl](../README.md)

What measuring this runtime on two Arc cards taught, most of it the hard way:

- **Bound the prize before building.** Make the cost free and time it: with every expert miss free decode was only
  13% faster (`NS_FREE_MISSES`), which ended a CPU-expert path that computed an expert twice as fast as its copy;
  simulate on a trace first (`reference/expert-cache`): pinning hot experts lost (the hot set follows the topic),
  filling VRAM by them at load won.
- **A microbenchmark lies three ways.** One expert reused stays in the L2 (rotate many); a constant fill is
  compressed by the GPU's memory (use random data); a naive reference kernel flatters the new one (the engine's
  own expansion was already 40% faster than the bench's). Confirm every win in the engine's own profile
  (`NS_PROFILE=gpu`, per GPU with `NS_PROFILE_PART`).
- **Exactness is the test that finds bugs.** Verify passes must equal one-token decode and batches each row alone,
  bit for bit (`spec-check`, `batch-check`) - and at long context too: 3-row passes drifted only past 2,048 tokens,
  where the indexer's 4-slot ring of keys was overwritten by rejected rows. A near-tie token flipping is not a bug;
  a growing difference is.
- **Two GPUs in turn is one GPU at a time.** Running the layers' halves as a pipeline (the first GPU a chunk ahead)
  read long prompts 1.4-1.7x faster; then the weaker card set the pace, and tuning per card (the KDA scan's columns
  by its compute units) mattered more than tuning per kernel.
- **Count every byte of VRAM, then guard it.** The expert budget kept a fixed margin while the forward pass's own
  buffers grew with the prompt chunk; past it the driver's spill path is the box's known hang. Budget them per
  chunk, stop before a GPU runs short (`NS_VRAM_GUARD_GIB`), and test VRAM changes through the server (its sessions
  are reserved), not only `generate`.
- **On an MoE, a row costs its experts.** A verify pass of 5 rows costs ~2.5x one of 2 (each token brings its own 8
  experts), so prompt-lookup drafts accepted 90% of the time still lost; raise the acceptance of the 2-row pass
  instead (speculative sampling).
- **Fusion pays where the decode is costly and shared.** Decoding 2-bit weights straight into the matrix units beat
  expand + oneMKL only once a work-group decoded each block once for all its threads (ESIMD, local memory) - and
  only for IQ2_XXS's table-driven gate | up: the down projection (Q2_K, no table, short rows) lost. Above ~256
  tokens oneMKL's tiles win anyway.
- **The host is part of the kernel.** Sorting the whole vocabulary to sample cost 6% of decode; one host wait on a
  router read a layer is fine while the GPU stays busy - check that it does (summed device time vs wall time).
- **Asynchronous means asynchronous.** A queued copy's host buffer must outlive it (a benchmark that freed it
  faulted the GPU); the host never waits on a queue's events (Level Zero v2 deadlocks - order on the device).
- **A long prompt must not hold the others.** Read it in chunks that stop for a newcomer, share the GPUs by time
  with those decoding, and keep its checkpoint (on disk) so the follow-up is seconds, not minutes.

- **Tune for one model in its own silo.** Speeding the 27B by editing Strata's shared kernels would have moved the
  engines tuned on them; the same changes as the 27B's own library (`kernels/llm/qwen35/silo`) gave it 1.3-4.3x per
  type and left the others bit for bit as they were (re-benched).
- **On Xe2, small loads are slow.** One- and two-byte global loads cost more than the arithmetic: the same IQ3_S
  kernel ran at 132 GB/s with them and 439 with aligned 16-byte loads and selects; 32-bit integer multiplies are slow
  too (a table instead). Codebooks belong in local memory, indexed by a whole byte.
- **Measure utilisation, not time.** A per-kernel profile against the card's ceilings (bytes and flops per kernel,
  device timestamps) showed decode attention at 8% of the bandwidth (36 work-groups on 256 engines), a norm walking
  its row on one core (22 us for 20 KB) and Q6_K's prompt expansion at 13 GB/s - none of it visible in the phase times.
- **Make the draft path exact first.** The decode attention sized its splits from the longest row; a verify row then
  summed its keys in another order than the one-token decode. Splitting each row as a one-row pass would made
  `spec-check` 0, and with it MTP's output the same text as without.
