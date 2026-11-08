#!/usr/bin/env python3
"""Compare two dumps of ripple_ref.py / NS_RC_DUMP=DIR (float32 files named embd, l_out-N, result_norm, logits):

    python3 -I compare.py --ref DIR --run DIR [--rows N]

Per tensor: cosine over all rows, the worst row's cosine, max |difference|, and the difference's RMS over the
reference's RMS. For logits also whether the largest logit is the same token and the KL(ref || run) of the softmaxes.
"""
import argparse
import glob
import os
import re

import numpy as np


def load(d, name):
    p = os.path.join(d, name + ".f32")
    return np.fromfile(p, dtype=np.float32) if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--width", type=int, default=5120)
    a = ap.parse_args()
    names = sorted((os.path.basename(p)[:-4] for p in glob.glob(os.path.join(a.ref, "*.f32"))),
                   key=lambda n: (0 if n == "embd" else 1 if n.startswith("l_out") else 2 if n == "result_norm" else 3,
                                  int(re.sub(r"\D", "", n) or 0)))
    worst = 1.0
    for n in names:
        r, x = load(a.ref, n), load(a.run, n)
        if x is None:
            print("%-14s missing in the run" % n)
            continue
        if n == "logits":
            lsm = lambda v: v - (v.max() + np.log(np.exp(v - v.max()).sum()))
            p, q = lsm(r.astype(np.float64)), lsm(x.astype(np.float64))
            kl = float((np.exp(p) * (p - q)).sum())
            print("%-14s argmax ref %d run %d  KL %.6f  max|d| %.4g  cos %.7f" % (n, r.argmax(), x.argmax(), kl, np.abs(r - x).max(),
                                                                                   float(r @ x / (np.linalg.norm(r) * np.linalg.norm(x)))))
            continue
        rows = min(r.size, x.size) // a.width
        if n != "result_norm" and r.size != x.size:
            # the run dumped every row of the pass, the reference the same prompt: sizes must agree
            print("%-14s sizes differ: ref %d run %d" % (n, r.size, x.size))
            continue
        R, X = r.reshape(-1, a.width), x.reshape(-1, a.width)
        cos = (R * X).sum(1) / (np.linalg.norm(R, axis=1) * np.linalg.norm(X, axis=1))
        worst = min(worst, float(cos.min()))
        rel = float(np.sqrt(((R - X) ** 2).mean() / (R ** 2).mean()))
        print("%-14s rows %-4d cos %.7f (worst row %.7f)  max|d| %.4g  rel RMS %.3g" % (n, rows, cos.mean(), cos.min(), np.abs(R - X).max(), rel))
    print("worst row cosine of any tensor: %.7f" % worst)


if __name__ == "__main__":
    main()
