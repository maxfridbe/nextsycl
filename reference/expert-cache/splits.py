"""Splits: MoE layers 3..k-1 on the first GPU (B65), k..44 on the second (B70); steady-state LRU misses per GPU
(run 0 warms, run 1 counted), slots moving ~23 per layer moved; PCIe seconds of copies up per token at a link speed."""
import collections, sys
seq = []   # (run, layer, experts) in file order
run = -1; seen_tick = {}
for line in open(sys.argv[1]):
    g, t, l, ex = line.split(); g, t, l = int(g), int(t), int(l)
    if g in seen_tick and t < seen_tick[g]: pass
    seen_tick[g] = t
    seq.append((l, [int(x) for x in ex.split(",")]))
# runs: the layer sequence restarts at 3 after the last layer each pass; split where a whole new process began
# (the trace's second run starts after the first's 700 tokens): find the index where ticks reset for GPU 1
ticks = []
for line in open(sys.argv[1]):
    g, t, l, _ = line.split(); ticks.append((int(g), int(t)))
cut = next(i for i in range(1, len(ticks)) if ticks[i][0] == 1 and any(tt[0] == 1 for tt in ticks[:i]) and ticks[i][1] < max(tt[1] for tt in ticks[:i] if tt[0] == 1))
warm, test = seq[:cut], seq[cut:]
TOTAL = 3784 + 3633; PER_LAYER = 23; SB = 6.8e6
passes = sum(1 for l, _ in test if l == 3)

def lru_misses(reqs_warm, reqs_test, cap, layers):
    c = collections.OrderedDict()
    init = [(l, e) for l in sorted(layers) for e in range(288)][:cap]
    for k in init: c[k] = None
    def go(reqs, count):
        m = 0
        for l, exs in reqs:
            if l not in layers: continue
            keys = [(l, e) for e in exs]
            for k in keys:
                if k in c: c.move_to_end(k)
            for k in keys:
                if k not in c:
                    m += count
                    if len(c) >= cap:
                        for v in c:
                            if v not in keys: del c[v]; break
                    c[k] = None
        return m
    go(reqs_warm, 0)
    return go(reqs_test, 1)

print(f"{passes} passes counted")
print(f"{'B65 layers':>10} {'B65 slots':>9} {'B70 slots':>9} {'miss65':>7} {'miss70':>7} | ms/pass copying up: today x8/x8 (25/25)  after x4/x16 (6.5/48)")
for k in (12, 15, 18, 22, 25):
    l65 = set(range(3, k)); l70 = set(range(k, 45))
    shift = (22 - k) * PER_LAYER          # today B65 holds 0..21
    c65, c70 = 3784 + shift, 3633 - shift
    m65 = lru_misses(warm, test, c65, l65); m70 = lru_misses(warm, test, c70, l70)
    today = (m65 / 25e9 + m70 / 25e9) * SB / passes * 1e3
    after = (m65 / 6.5e9 + m70 / 48e9) * SB / passes * 1e3
    print(f"{k:>10} {c65:>9} {c70:>9} {m65:>7} {m70:>7} |   {today:6.1f}                       {after:6.1f}")
