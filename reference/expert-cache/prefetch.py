"""Prefetch rules on the guess trace, with an LRU cache per GPU (B65 layers 0-21: 3784 slots; B70 22-44: 3633):
per rule, the misses it covers (fetched ahead) and the copies it wastes (fetched, then not asked for)."""
import collections, sys
H1 = [0.96, 0.92, 0.86, 0.77, 0.66, 0.56, 0.45, 0.36, 0.29, 0.22, 0.18, 0.15, 0.12, 0.11, 0.09, 0.08]
H2 = [0.93, 0.84, 0.75, 0.64, 0.52, 0.44, 0.36, 0.29, 0.24, 0.21, 0.18, 0.16, 0.13, 0.12, 0.10, 0.10]
CAP = lambda l: 3784 if l < 22 else 3633
lines = [l.split() for l in open(sys.argv[1])]

def sim(rule):
    caches = {0: collections.OrderedDict(), 1: collections.OrderedDict()}
    gpu = lambda l: 0 if l < 22 else 1
    for g in (0, 1):   # the load-time layout: layers in order
        layers = range(3, 22) if g == 0 else range(22, 45)
        for k in [(l, e) for l in layers for e in range(288)][:CAP(layers[0])]:
            caches[g][k] = None
    guesses = collections.defaultdict(list)   # target layer -> [(ahead, row, ranked)]
    fetched = {}                              # (layer, expert) -> still unused
    covered = misses = wasted = issued = 0
    def insert(k, pinned):
        c = caches[gpu(k[0])]
        if k in c: c.move_to_end(k); return
        if len(c) >= CAP(k[0]):
            for v in c:
                if v not in pinned:
                    del c[v]
                    if fetched.pop(v, False): pass
                    break
        c[k] = None
    for f in lines:
        if f[0] == "G":
            guesses[int(f[2])].append((int(f[2]) - int(f[1]), int(f[3]), [int(x) for x in f[4].split(",")]))
            continue
        l, r = int(f[1]), int(f[2])
        ex = [int(x) for x in f[3].split(",")] if len(f) > 3 else []
        if r == 0:
            # the layer's request starts: apply the rule to its guesses (made a layer or two earlier)
            g = guesses.pop(l, [])
            for e in rule(g):
                k = (l, e)
                if k not in caches[gpu(l)]:
                    issued += 1; fetched[k] = True; insert(k, set())
        keys = [(l, e) for e in ex]
        for k in keys:
            c = caches[gpu(l)]
            if k in c:
                if fetched.pop(k, False): covered += 1
                c.move_to_end(k)
            else:
                misses += 1; insert(k, set(keys))
    wasted = issued - covered
    return covered, misses, issued, wasted

def p_used(g, ahead, H):
    p = collections.defaultdict(lambda: 1.0)
    for a, r, ranked in g:
        if a != ahead: continue
        for i, e in enumerate(ranked):
            p[e] *= 1 - H[i]
    return {e: 1 - q for e, q in p.items()}

rules = {"none": lambda g: []}
for th in (0.9, 0.8, 0.6, 0.4):
    rules[f"1 ahead, p>={th}"] = (lambda th: lambda g: [e for e, p in p_used(g, 1, H1).items() if p >= th])(th)
for th in (0.8, 0.6):
    rules[f"2 ahead, p>={th}"] = (lambda th: lambda g: [e for e, p in p_used(g, 2, H2).items() if p >= th])(th)
for name, rule in rules.items():
    cov, mis, iss, was = sim(rule)
    print(f"{name:18} demand misses {mis:6}  fetched ahead {iss:6}  of them used {cov:6}  wasted {was:6}")
