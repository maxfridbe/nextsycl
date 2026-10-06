"""Score the router guesses: for each guess (from layer l, for layer l+a, row r, ranked), the next actual request of
that layer and row; hit rate at each rank, and the share of the actual experts the top-k covers."""
import collections, sys
lines = [l.split() for l in open(sys.argv[1])]
pending = collections.defaultdict(list)   # (target layer, row) -> guesses waiting
hit = {1: [0] * 16, 2: [0] * 16}; n = {1: 0, 2: 0}; cover = {1: collections.Counter(), 2: collections.Counter()}
for f in lines:
    if f[0] == "G":
        l, tl, r, ex = int(f[1]), int(f[2]), int(f[3]), [int(x) for x in f[4].split(",")]
        pending[(tl, r)].append((tl - l, ex))
    else:
        l, r, ex = int(f[1]), int(f[2]), set(int(x) for x in f[3].split(",")) if len(f) > 3 else set()
        for a, g in pending.pop((l, r), []):
            n[a] += 1
            for i, e in enumerate(g):
                hit[a][i] += e in ex
            for k in (1, 2, 4, 8, 12, 16):
                cover[a][k] += len(set(g[:k]) & ex)
for a in (1, 2):
    print(f"{a} layer(s) ahead, {n[a]} guesses: hit rate by rank " + " ".join(f"{h / max(n[a], 1):.2f}" for h in hit[a]))
    print(f"   of the 8 actual experts, the top k cover: " + "  ".join(f"k={k}: {cover[a][k] / max(n[a], 1):.2f}" for k in (1, 2, 4, 8, 12, 16)))
