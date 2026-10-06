# Expert cache studies

Offline simulations on a decode trace (`NS_TRACE_EXPERTS=FILE nextsycl generate ...`: one line per layer request,
`GPU tick layer experts`). Slot counts are in the scripts (from `nextsycl status`).

- `policies.py TRACE`: misses per policy per run (LRU, LFU with decays, Belady's optimum)
- `steady.py policies.py TRACE`: the same with run 0 warming the cache, run 1 counted
- `admission.py policies.py TRACE`: LRU that only keeps an expert on its second miss (ghost list)
- `splits.py TRACE`: misses per GPU for layer splits, and the PCIe time of the copies at given link speeds

Findings (2026-10-06, 1,200 tokens, two topics): steady-state LRU is the best practical policy - LFU loses (the
topic shifts), admission halves the copies down but adds 10-15% copies up (the ones decode waits on); Belady's
optimum has 3x fewer misses, reachable only by prediction (the prefetch). With the B70 on x16 and the B65 on x4, the
split to use is NS_SPLIT=15: the B65's 12 MoE layers all resident (no swaps over its slow link), the copy time a
pass ~2/3 of today's best.
