"""LRU with admission: a miss is admitted to VRAM only when it missed before within the ghost window (else it is
staged for one use, evicting nothing). Transfers up = every miss; copies down = admitted ones."""
import collections, sys
src = open(sys.argv[1]).read().split("for (g, r), reqs in sorted")[0]; sys.argv = [sys.argv[0], sys.argv[2]]; exec(src)

class Admit:
    def __init__(s, g, reqs, ghost):
        s.c = collections.OrderedDict((k, None) for k in initial(g, reqs)); s.cap = CAP[g]; s.miss = 0; s.down = 0
        s.ghost = collections.OrderedDict(); s.G = ghost
    def access(s, l, exs):
        keys = [(l, e) for e in exs]
        for k in keys:
            if k in s.c: s.c.move_to_end(k)
        for k in keys:
            if k in s.c: continue
            s.miss += 1
            if s.G and k not in s.ghost:
                s.ghost[k] = None
                if len(s.ghost) > s.G: s.ghost.popitem(last=False)
                continue                       # staged: used once, not kept
            s.ghost.pop(k, None)
            if len(s.c) >= s.cap:
                for v in s.c:
                    if v not in keys: del s.c[v]; s.down += 1; break
            s.c[k] = None

for g in (1, 0):
    warm, test = runs[(g, 0)], runs[(g, 1)]
    row = []
    for G in (0, 500, 2000, 5000, 10000, 20000):
        st = Admit(g, warm + test, G)
        for l, exs in warm: st.access(l, exs)
        st.miss = st.down = 0
        for l, exs in test: st.access(l, exs)
        row.append(f"ghost {G}: up {st.miss} down {st.down}")
    print(f"GPU {g}: " + " | ".join(row))
