"""Two run dumps of ref.py compared tensor by tensor (the taps, every step's estimates, the finished latents, the frames):
how far a change at one place carries - e.g. `--perturb` (noise the size of the engine's error after block 0) against
the run without, for the base and for an adapter.

    python -I cmp.py A.safetensors B.safetensors
"""
import sys

import torch
from safetensors.torch import load_file


def main():
    a, b = load_file(sys.argv[1]), load_file(sys.argv[2])
    keys = sorted(set(a) & set(b), key=lambda k: (not k.startswith('tap.'), int(k[4:]) if k.startswith('tap.') else 0, k))
    for k in keys:
        x, y = a[k].double().flatten(), b[k].double().flatten()
        if x.shape != y.shape or k in ('context', 'noise.video', 'noise.audio', 'sigmas'):
            continue
        rel = float((x - y).norm() / y.norm().clamp_min(1e-30))
        cos = float(torch.nn.functional.cosine_similarity(x, y, 0))
        extra = ''
        if k == 'images':
            mse = float(((x - y) ** 2).mean())
            extra = f'  PSNR {10 * torch.log10(torch.tensor(1 / max(mse, 1e-20))).item():.1f} dB'
        print(f'{k:22s} rel {rel:.3e}  cosine {cos:.6f}{extra}')


if __name__ == '__main__':
    main()
