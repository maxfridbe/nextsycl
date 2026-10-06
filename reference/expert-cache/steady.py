"""Steady state: run 0 warms the cache, misses counted on run 1 only."""
import collections, sys, math
src = open(sys.argv[1]).read().split("for (g, r), reqs in sorted")[0]; sys.argv = [sys.argv[0], sys.argv[2]]; exec(src)   # load runs + CAP/BASE/initial from sim.py

def run_policy(g, warm, test, make):
    st = make(g, warm + test)
    for l, exs in warm:
        st.access(l, exs)
    st.miss = 0
    for l, exs in test:
        st.access(l, exs)
    return st.miss

class LRU:
    def __init__(s, g, reqs):
        s.c = collections.OrderedDict((k, None) for k in initial(g, reqs)); s.cap = CAP[g]; s.miss = 0
    def access(s, l, exs):
        keys = [(l, e) for e in exs]
        for k in keys:
            if k in s.c: s.c.move_to_end(k)
        for k in keys:
            if k not in s.c:
                s.miss += 1
                if len(s.c) >= s.cap:
                    for v in s.c:
                        if v not in keys: del s.c[v]; break
                s.c[k] = None

class LFU:
    def __init__(s, g, reqs, decay):
        s.c = set(initial(g, reqs)); s.cap = CAP[g]; s.miss = 0; s.sc = collections.defaultdict(float); s.w = 1.0; s.d = decay
    def access(s, l, exs):
        s.w /= s.d
        keys = [(l, e) for e in exs]
        for k in keys: s.sc[k] += s.w
        for k in keys:
            if k not in s.c:
                s.miss += 1
                if len(s.c) >= s.cap:
                    v = min((x for x in s.c if x not in keys), key=lambda x: s.sc[x]); s.c.discard(v)
                s.c.add(k)

class Belady:
    def __init__(s, g, reqs):
        s.reqs = reqs; s.i = 0; s.c = set(initial(g, reqs)); s.cap = CAP[g]; s.miss = 0
        nxt = {}; s.fut = [None] * len(reqs)
        for i in range(len(reqs) - 1, -1, -1):
            l, exs = reqs[i]; s.fut[i] = {}
            for e in exs: s.fut[i][(l, e)] = nxt.get((l, e), 1 << 60); nxt[(l, e)] = i
        s.nu = dict(nxt)
    def access(s, l, exs):
        keys = [(l, e) for e in exs]
        for k in keys: s.nu[k] = s.fut[s.i][k]
        for k in keys:
            if k not in s.c:
                s.miss += 1
                if len(s.c) >= s.cap:
                    v = max((x for x in s.c if x not in keys), key=lambda x: s.nu.get(x, 1 << 61)); s.c.discard(v)
                s.c.add(k)
        s.i += 1

for g in (1, 0):
    warm, test = runs[(g, 0)], runs[(g, 1)]
    out = {"LRU": run_policy(g, warm, test, LRU)}
    for d in (0.999, 0.9995, 0.9999, 1.0):
        out[f"LFU{d}"] = run_policy(g, warm, test, lambda g, r: LFU(g, r, d))
    out["Belady"] = run_policy(g, warm, test, Belady)
    print(f"GPU {g}: run-1 misses  " + "  ".join(f"{k} {v}" for k, v in out.items()))
