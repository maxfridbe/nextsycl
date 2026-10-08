"""Pinning by a usage profile on a recorded decode trace: a fraction of each GPU's slots holds the experts used most in
ANOTHER run (the profile: another topic), never evicted; the rest LRU. Misses on the test run against plain LRU, both
warmed by the profile run first (steady state). Usage: pinning.py policies.py TRACE"""
import collections, sys
sys.argv = [sys.argv[0], sys.argv[2]] if len(sys.argv) > 2 else sys.argv
exec(open(sys.argv[0].replace("pinning.py", "policies.py")).read().split("def lru")[0])   # the trace, CAP, BASE

def run_cache(g, warm, test, pinned):
    cap = CAP[g] - len(pinned)
    cache = collections.OrderedDict()
    miss = 0
    for phase, reqs in (("warm", warm), ("test", test)):
        for l, exs in reqs:
            keys = [(l, e) for e in exs]
            for k in keys:
                if k in cache:
                    cache.move_to_end(k)
            for k in keys:
                if k in pinned or k in cache:
                    continue
                if phase == "test":
                    miss += 1
                if len(cache) >= cap:
                    for v in cache:
                        if v not in keys:
                            del cache[v]; break
                cache[k] = None
    return miss

for g in sorted({g for g, _ in runs}):
    rs = sorted(r for gg, r in runs if gg == g)
    if len(rs) < 2:
        continue
    for prof, test in ((rs[0], rs[1]), (rs[1], rs[0])):
        counts = collections.Counter(k for l, exs in runs[(g, prof)] for k in ((l, e) for e in exs))
        ranked = [k for k, _ in counts.most_common()]
        line = []
        for frac in (0.0, 0.25, 0.5, 0.75, 0.9):
            pinned = set(ranked[: int(CAP[g] * frac)])
            m = run_cache(g, runs[(g, prof)], runs[(g, test)], pinned)
            line.append(f"{int(frac * 100)}%: {m}")
        n = sum(len(e) for _, e in runs[(g, test)])
        print(f"GPU {g}, profile run {prof} -> test run {test} ({n} requests): misses with pinned " + ", ".join(line))

# Cold start: the cache at load filled in layer order (today) or with the profile run's most used, then plain LRU on
# the test run - the first answer after a restart
def cold(g, test, initial_keys):
    cache = collections.OrderedDict((k, None) for k in initial_keys[: BASE[g]])
    miss = 0
    for l, exs in test:
        keys = [(l, e) for e in exs]
        for k in keys:
            if k in cache:
                cache.move_to_end(k)
        for k in keys:
            if k not in cache:
                miss += 1
                if len(cache) >= CAP[g]:
                    for v in cache:
                        if v not in keys:
                            del cache[v]; break
                cache[k] = None
    return miss

print("cold start (first answer after a load):")
for g in sorted({g for g, _ in runs}):
    rs = sorted(r for gg, r in runs if gg == g)
    if len(rs) < 2:
        continue
    for prof, test in ((rs[0], rs[1]), (rs[1], rs[0])):
        counts = collections.Counter(k for l, exs in runs[(g, prof)] for k in ((l, e) for e in exs))
        ranked = [k for k, _ in counts.most_common()]
        layer_order = initial(g, runs[(g, test)])
        # the profile's ranking, then the rest in layer order
        seen = set(ranked)
        prof_order = ranked + [k for k in layer_order if k not in seen]
        # the first 64 tokens' worth of requests (a short answer) and the whole run
        t = runs[(g, test)]
        first = t[: len(t) // 10]
        print(f"  GPU {g}, profile {prof} -> test {test}: first tenth {cold(g, first, layer_order)} vs {cold(g, first, prof_order)} misses; "
              f"whole run {cold(g, t, layer_order)} vs {cold(g, t, prof_order)} (layer order vs profile)")
