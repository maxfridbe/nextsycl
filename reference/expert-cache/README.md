# Expert cache studies

Offline simulations on a decode trace (`NS_TRACE_EXPERTS=FILE nextsycl generate ...`: one line per layer request,
`GPU tick layer experts`). Slot counts are in the scripts (from `nextsycl status`).

- `policies.py TRACE`: misses per policy per run (LRU, LFU with decays, Belady's optimum)
- `steady.py policies.py TRACE`: the same with run 0 warming the cache, run 1 counted
- `admission.py policies.py TRACE`: LRU that only keeps an expert on its second miss (ghost list)
- `guesses.py GUESS_TRACE` (`NS_TRACE_GUESS=FILE`: the next two layers' routers on each layer's input, and what
  each layer asked for): the guesses' hit rate by rank, one and two layers ahead
- `prefetch.py GUESS_TRACE`: prefetch rules by chance of use (from the rank hit rates) against an LRU cache
- `splits.py TRACE`: misses per GPU for layer splits, and the PCIe time of the copies at given link speeds

Findings (2026-10-06, 1,200 tokens, two topics): steady-state LRU is the best practical policy - LFU loses (the
topic shifts), admission halves the copies down but adds 10-15% copies up (the ones decode waits on); Belady's
optimum has 3x fewer misses, reachable only by prediction (the prefetch). With the B70 on x16 and the B65 on x4, the
split to use is NS_SPLIT=15: the B65's 12 MoE layers all resident (no swaps over its slow link), the copy time a
pass ~2/3 of today's best.

Guesses (2026-10-06, 300 tokens): the next layer's router on this layer's input has its best expert right 96% of the
time, its 4th 77%, its 8th 36%; two layers ahead 93% / 64% / 29%. Fetching every non-resident expert with a chance
of use >= 0.6 (the rank hit rates combined over a pass's rows) is the measured best: decode 18.4 -> 19.4 tok/s - once
the guess is read at the router's own wait (read after the layer's expert work was queued, it made the GPU idle).
