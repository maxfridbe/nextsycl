#!/usr/bin/env python3
"""Fetch only the tensors a layer-wise check needs from ricursiveAI/Ripple-65B-Preview, by HTTP range, so none of the
16 shards (130 GB) is downloaded whole: the embeddings, the first N blocks, the output norm and the head.

    python3 -I fetch_tensors.py --revision REV --layers 3 --out DIR [--no-head]

Writes DIR/<tensor name>.bf16 (raw little-endian bf16) and DIR/index.json (name -> shape). The bytes are data, read
back by ripple_ref.py; nothing downloaded is executed. Run with the repository revision pinned (the card's).
"""
import argparse
import json
import os
import struct
import urllib.request

REPO = "ricursiveAI/Ripple-65B-Preview"


def get(url, rng=None):
    req = urllib.request.Request(url)
    if rng:
        req.add_header("Range", "bytes=%d-%d" % rng)
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revision", default="main")
    ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-head", action="store_true", help="skip lm_head and the final norm")
    a = ap.parse_args()
    base = "https://huggingface.co/%s/resolve/%s/" % (REPO, a.revision)
    index = json.loads(get(base + "model.safetensors.index.json"))["weight_map"]
    want = ["model.embed_tokens.weight"]
    if not a.no_head:
        want += ["model.norm.weight", "lm_head.weight"]
    for l in range(a.layers):
        want += sorted(k for k in index if k.startswith("model.layers.%d." % l))
    os.makedirs(a.out, exist_ok=True)
    meta_path = os.path.join(a.out, "index.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    headers = {}
    for name in want:
        dst = os.path.join(a.out, name + ".bf16")
        if name in meta and os.path.exists(dst):
            continue
        shard = index[name]
        if shard not in headers:
            n = struct.unpack("<Q", get(base + shard, (0, 7)))[0]
            headers[shard] = (n, json.loads(get(base + shard, (8, 8 + n - 1))))
        n, h = headers[shard]
        t = h[name]
        if t["dtype"] != "BF16":
            raise SystemExit("%s: %s, not BF16" % (name, t["dtype"]))
        lo, hi = t["data_offsets"]
        start = 8 + n + lo
        with open(dst + ".part", "wb") as f:
            # 64 MiB ranges
            at = start
            while at < start + (hi - lo):
                end = min(at + (64 << 20), start + (hi - lo)) - 1
                f.write(get(base + shard, (at, end)))
                at = end + 1
        os.replace(dst + ".part", dst)
        meta[name] = t["shape"]
        json.dump(meta, open(meta_path, "w"))
        print("fetched", name, t["shape"], hi - lo, "bytes", flush=True)


if __name__ == "__main__":
    main()
