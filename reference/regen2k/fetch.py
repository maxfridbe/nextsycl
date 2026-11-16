"""Some tensors of a remote safetensors checkpoint, by HTTP byte range, into one local safetensors - so the parity
checks of Open-H3-Regenerate-2K-IC read the base model's time embedder, the blocks' full AdaLN projections or one
matrix without downloading the whole checkpoint.

    python -I fetch.py --url https://huggingface.co/<repo>/resolve/<rev>/<file>.safetensors \
        --match '^time_embedder\\.' --match '^blocks\\.[0-9]+\\.adaln_proj\\.' --out part.safetensors
    python -I fetch.py --url https://huggingface.co/<repo>/resolve/<rev>/<dir>/model.safetensors.index.json ...   (sharded)

The file is written as `nextsycl models pull` writes a card's `tensors` part (glue/models/src/catalog.rs
fetch_tensors): the header compact with its keys sorted, the source's metadata kept, the data in the source's order -
so its SHA-256 is the card's. The token is huggingface_hub's (HF_TOKEN, else ~/.cache/huggingface/token); never printed.
"""
import argparse
import json
import os
import re
import struct
import urllib.request
from pathlib import Path


def token():
    t = os.environ.get('HF_TOKEN')
    if t:
        return t.strip()
    p = Path(os.path.expanduser('~/.cache/huggingface/token'))
    return p.read_text().strip() if p.exists() else None


def get(url, tok, start=None, length=None):
    headers = {'Accept-Encoding': 'identity'}
    if start is not None:
        headers['Range'] = f'bytes={start}-{start + length - 1}'
    if tok:
        headers['Authorization'] = f'Bearer {tok}'
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=600) as r:
        if start is not None and r.status != 206:
            raise RuntimeError('the server ignored the Range request')
        data = r.read()
    if start is not None and len(data) != length:
        raise RuntimeError(f'{url}: {len(data)} of {length} bytes')
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', required=True, help='a .safetensors file, or a sharded checkpoint\'s .index.json')
    ap.add_argument('--match', action='append', required=True, help='regex over tensor names (any matches)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    tok = token()
    pats = [re.compile(m) for m in a.match]
    if a.url.endswith('.index.json'):
        index = json.loads(get(a.url, tok))
        folder = a.url.rsplit('/', 1)[0] + '/'
        shards = sorted({folder + s for n, s in index['weight_map'].items() if any(p.search(n) for p in pats)})
    else:
        shards = [a.url]
    keep, meta, offset = [], None, 0
    for url in shards:
        n, = struct.unpack('<Q', get(url, tok, 0, 8))
        h = json.loads(get(url, tok, 8, n))
        meta = h.pop('__metadata__', None) if meta is None else meta
        for name, e in sorted(h.items(), key=lambda kv: kv[1]['data_offsets'][0]):
            if name != '__metadata__' and any(p.search(name) for p in pats):
                keep.append((url, 8 + n, name, e))
    header = {}
    for url, base, name, e in keep:
        s, t = e['data_offsets']
        header[name] = {'data_offsets': [offset, offset + t - s], 'dtype': e['dtype'], 'shape': e['shape']}
        offset += t - s
    if meta is not None:
        header['__metadata__'] = dict(sorted(meta.items()))
    raw = json.dumps(dict(sorted(header.items())), separators=(',', ':'), ensure_ascii=False).encode()
    raw += b' ' * ((8 - len(raw) % 8) % 8)
    with open(a.out, 'wb') as f:
        f.write(struct.pack('<Q', len(raw)))
        f.write(raw)
        for url, base, name, e in keep:
            s, t = e['data_offsets']
            while s < t:
                n = min(t - s, 64 << 20)
                f.write(get(url, tok, base + s, n))
                s += n
            print(name, e['dtype'], e['shape'], flush=True)


if __name__ == '__main__':
    main()
