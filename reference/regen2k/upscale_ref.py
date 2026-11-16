"""The latent upscaler as Open-H3-Regenerate-2K-IC runs it (infer/ref2va_ic/upscale.py: the author's node, target
dimensions, the whole clip in one piece) on the draft latent the engine encoded (`z` of a regen2k check dump), against
the engine's start latent (`up`).

    python -I upscale_ref.py --code <release>/infer --upscaler minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors --dump check.safetensors
"""
import argparse
import sys

import torch
from safetensors.torch import load_file


def main():
    ap = argparse.ArgumentParser()
    for k in ('code', 'upscaler', 'dump'):
        ap.add_argument('--' + k, required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.code)
    from ref2va_ic import upscale
    node = upscale.load(a.upscaler)
    d = load_file(a.dump)
    z, ours = d['z'].float(), d['up'].float()
    out = upscale._pass(node, z, list(ours.shape), 'cpu').float()
    rel = float((ours - out).norm() / out.norm())
    cos = float(torch.nn.functional.cosine_similarity(ours.double().flatten(), out.double().flatten(), 0))
    print(f'upscaled {tuple(z.shape)} -> {tuple(out.shape)}: engine vs the release rel {rel:.4f}, cosine {cos:.6f}')


if __name__ == '__main__':
    main()
