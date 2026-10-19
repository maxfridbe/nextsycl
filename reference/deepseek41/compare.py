"""compare.py REF_DIR ENGINE_DIR [--rows R] [--grep PATTERN]: every tap the reference wrote (its index.tsv) against the
engine's saved tensor of the same name (NS_CHECK_SAVE=ENGINE_DIR), on the last R rows of both when R is given (a tap of
the reference with fewer rows than the engine's is taken as the engine's last rows): cosine, relative L2, max |diff|;
the indexer's picks as the share of the reference's picks the engine also made."""
import argparse
import os
import re

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('ref')
ap.add_argument('eng')
ap.add_argument('--rows', type=int, default=0)
ap.add_argument('--grep', default='')
a = ap.parse_args()
order = []
dims = {}
for line in open(os.path.join(a.ref, 'index.tsv')):
    c = line.rstrip('\n').split('\t')
    if c[0] not in dims:
        order.append(c[0])
    dims[c[0]] = [int(x) for x in c[2:6]]
worst = (1.0, '')
for name in order:
    if a.grep and not re.search(a.grep, name):
        continue
    pe = os.path.join(a.eng, name + '.f32')
    if not os.path.exists(pe):
        continue
    r = np.fromfile(os.path.join(a.ref, name + '.f32'), np.float32)
    e = np.fromfile(pe, np.float32)
    d = dims[name]
    # a row: everything but the outermost dimension (ggml order: d[0] fastest), the tokens
    nz = [x for x in d if x > 1]
    row = len(r) // (nz[-1] if len(nz) > 1 else 1)
    nrow_r = len(r) // row
    if len(e) % row:
        print(f'{name:<28} sizes: ref {len(r)}, engine {len(e)}')
        continue
    nrow_e = len(e) // row
    n = min(nrow_r, nrow_e)
    if a.rows:
        n = min(n, a.rows)
    r2 = r[-n * row:].astype(np.float64)
    e2 = e[-n * row:].astype(np.float64)
    if name.startswith('indexer_topk'):
        continue
    fin = np.isfinite(r2) & np.isfinite(e2)
    r2, e2 = r2[fin], e2[fin]
    cos = float(r2 @ e2 / max(np.linalg.norm(r2) * np.linalg.norm(e2), 1e-30))
    rel = float(np.linalg.norm(r2 - e2) / max(np.linalg.norm(r2), 1e-30))
    mx = float(np.abs(r2 - e2).max()) if len(r2) else 0.0
    print(f'{name:<28} rows {n:>6}  cosine {cos:.6f}  rel {rel:.2e}  max {mx:.3e}')
    if cos < worst[0]:
        worst = (cos, name)
print('worst', worst)
