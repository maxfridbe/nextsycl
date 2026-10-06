"""Expert cache policies on a recorded decode trace (GPU, tick, layer, experts): misses per policy."""
import collections, sys
CAP = {1: 3784, 0: 3633}      # slots per GPU (base + lent arena), from `nextsycl status`
BASE = {1: 3445, 0: 3316}     # filled at load (layer order)
runs = collections.defaultdict(list)   # (gpu, run) -> [(layer, [experts])]
last = {}
run = collections.Counter()
for line in open(sys.argv[1]):
    g, t, l, ex = line.split()
    g, t, l = int(g), int(t), int(l)
    if g in last and t < last[g]:
        run[g] += 1
    last[g] = t
    runs[(g, run[g])].append((l, [int(x) for x in ex.split(",")]))

def initial(g, reqs):
    layers = sorted({l for l, _ in reqs})
    return [(l, e) for l in layers for e in range(288)][:BASE[g]]

def lru(g, reqs):
    cache = collections.OrderedDict((k, None) for k in initial(g, reqs))
    miss = 0
    for l, exs in reqs:
        keys = [(l, e) for e in exs]
        for k in keys:
            if k in cache:
                cache.move_to_end(k)
        for k in keys:
            if k not in cache:
                miss += 1
                if len(cache) >= CAP[g]:
                    for v in cache:          # the least recent not used in this request
                        if v not in keys:
                            del cache[v]; break
                cache[k] = None
    return miss

def lfu(g, reqs, decay=0.999):
    cache = set(initial(g, reqs)); score = collections.defaultdict(float); miss = 0
    w = 1.0
    for l, exs in reqs:
        w /= decay                            # aging by growing the weight of new uses
        keys = [(l, e) for e in exs]
        for k in keys:
            score[k] += w
        for k in keys:
            if k not in cache:
                miss += 1
                if len(cache) >= CAP[g]:
                    v = min((x for x in cache if x not in keys), key=lambda x: score[x])
                    cache.discard(v)
                cache.add(k)
    return miss

def belady(g, reqs):
    nxt = {}; future = [None] * len(reqs)
    for i in range(len(reqs) - 1, -1, -1):
        l, exs = reqs[i]
        future[i] = {}
        for e in exs:
            future[i][(l, e)] = nxt.get((l, e), 1 << 60)
            nxt[(l, e)] = i
    nextuse = dict(nxt)      # first use of each key
    cache = set(initial(g, reqs)); miss = 0
    import heapq
    for i, (l, exs) in enumerate(reqs):
        keys = [(l, e) for e in exs]
        for k in keys:
            nextuse[k] = future[i][k]
        for k in keys:
            if k not in cache:
                miss += 1
                if len(cache) >= CAP[g]:
                    v = max((x for x in cache if x not in keys), key=lambda x: nextuse.get(x, 1 << 61))
                    cache.discard(v)
                cache.add(k)
    return miss

for (g, r), reqs in sorted(runs.items()):
    toks = len(reqs) / len({l for l, _ in reqs})
    res = {"LRU": lru(g, reqs), "LFU.999": lfu(g, reqs), "LFU.99": lfu(g, reqs, 0.99), "Belady": belady(g, reqs)}
    print(f"GPU {g} run {r}: {len(reqs)} layer requests (~{toks:.0f} passes)  " + "  ".join(f"{k} {v}" for k, v in res.items()))
