"""Is the int8 FL2VA denoiser we serve the base the pi-H3 adapters were trained on (multimodalart/MiniMax-H3-Pruned)?

A few tensors range-fetched from the Diffusers base's transformer (bf16) and compared with ours un-rotated (W = (q s) H per
256 group): the AdaLN curve, block 0's AdaLN projection and out matrix, a late block's fc2 - so the comparison is of the
weights, the difference only int8's rounding.

    python -I base_check.py --int8 minimax_h3_fl2va_pruned_int8_convrot.safetensors [--revision SHA]
"""
import argparse
import json
import struct
import urllib.request

import torch
from safetensors import safe_open

REPO = 'https://huggingface.co/multimodalart/MiniMax-H3-Pruned/resolve/{rev}/transformer/{file}'
DTYPES = {'BF16': torch.bfloat16, 'F16': torch.float16, 'F32': torch.float32}


def get(url, start=None, end=None):
    req = urllib.request.Request(url)
    if start is not None:
        req.add_header('Range', f'bytes={start}-{end}')
    with urllib.request.urlopen(req) as r:
        return r.read()


def hadamard(g=256):
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32)
    m = torch.ones(1, 1)
    while m.shape[0] < g:
        m = torch.kron(m, h4)
    return m / (g ** 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--int8', required=True)
    ap.add_argument('--revision', default='1a0ef5e65b639e84af81d883817968532180e9c7')
    a = ap.parse_args()
    index = json.loads(get(REPO.format(rev=a.revision, file='diffusion_pytorch_model.safetensors.index.json')))['weight_map']
    headers = {}

    def remote(name):
        f = index[name]
        url = REPO.format(rev=a.revision, file=f)
        if f not in headers:
            n = struct.unpack('<Q', get(url, 0, 7))[0]
            headers[f] = (n, json.loads(get(url, 8, 8 + n - 1)))
        n, h = headers[f]
        e = h[name]
        s, t = e['data_offsets']
        raw = get(url, 8 + n + s, 8 + n + t - 1)
        return torch.frombuffer(bytearray(raw), dtype=DTYPES[e['dtype']]).view(e['shape']).float()

    ours = safe_open(a.int8, 'pt')
    H = hadamard()

    def linear(name):
        w = ours.get_tensor(name + '.weight')
        if w.dtype != torch.int8:
            return w.float()
        q = w.float() * ours.get_tensor(name + '.weight_scale').float()
        n, k = q.shape
        return (q.view(n, k // 256, 256) @ H).view(n, k)

    def cmp(what, x, y):
        rel = float((x - y).norm() / y.norm())
        cos = float(torch.nn.functional.cosine_similarity(x.double().flatten(), y.double().flatten(), 0))
        print(f'{what:60s} rel {rel:.5f}  cosine {cos:.7f}  {tuple(y.shape)}', flush=True)

    print('their time / table keys:', [k for k in index if 'time' in k or 'table' in k or 'adaln_' in k.split('.')[0]][:8], flush=True)
    if 'time_embedder.table' in index:
        cmp('time_embedder.table vs adaln_t_table', ours.get_tensor('adaln_t_table').float(), remote('time_embedder.table'))
    cmp('transformer_blocks.0.adaln_proj.linear.weight', ours.get_tensor('blocks.0.adaln_proj.linear.weight').float(),
        remote('transformer_blocks.0.adaln_proj.linear.weight'))
    cmp('transformer_blocks.0.attn.to_out.0.weight (int8 rounding)', linear('blocks.0.attn.out_proj'), remote('transformer_blocks.0.attn.to_out.0.weight'))
    cmp('transformer_blocks.30.ff.net.2.weight (int8 rounding)', linear('blocks.30.mlp.fc2'), remote('transformer_blocks.30.ff.net.2.weight'))
    cmp('transformer_blocks.49.ff.net.2.weight (int8 rounding)', linear('blocks.49.mlp.fc2'), remote('transformer_blocks.49.ff.net.2.weight'))
    cmp('proj_out.weight vs final_layer.video_out', ours.get_tensor('final_layer.video_out.weight').float(), remote('proj_out.weight'))


if __name__ == '__main__':
    main()
