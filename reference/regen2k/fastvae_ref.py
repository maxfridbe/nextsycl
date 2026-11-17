"""Open-H3-Regenerate-2K's fast decoder as the release runs it (infer/ref2va_ic/fast_vae/decoder.py: NonCausalViTDecoder,
the plain PyTorch engine "ref"), on the CPU in float32, on a crop of real latents - against the engine's fast decoder
(crates/h3/src/fastvae.rs) through the decode job's check.

    python -I fastvae_ref.py --code <release>/infer --weights <release>/vae_fast/fast_h3_vae_v2_24l.safetensors \
        --config <release>/base_model/vae/config.json --latents regen1.latents.safetensors --crop 0,7,30,20,40,36 \
        --out-latents crop.safetensors --out-dump ref.safetensors

--crop t0,T,y0,H,x0,W in latent frames / pixels (t0 a block's first: a multiple of 5). Writes the crop as
`samples.video` [1, 24, T, H, W] (what the decode job reads) and the reference frames as `images` [F, 16H, 16W, 3] in
[0, 1] (what its check compares).
"""
import argparse
import json
import sys
import time

import torch
from safetensors.torch import load_file, save_file


def main():
    ap = argparse.ArgumentParser()
    for k in ('code', 'weights', 'config', 'latents', 'crop', 'out_latents', 'out_dump'):
        ap.add_argument('--' + k.replace('_', '-'), dest=k, required=True)
    ap.add_argument('--threads', type=int, default=24)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    sys.path.insert(0, a.code)
    from ref2va_ic.fast_vae.decoder import KEEP, NonCausalViTDecoder
    t0, T, y0, H, x0, W = (int(v) for v in a.crop.split(','))
    assert t0 % 5 == 0, 'the crop must start on a block (a multiple of 5 latent frames)'
    lat = load_file(a.latents)
    key = 'samples.video' if 'samples.video' in lat else 'latents.video'
    z = lat[key].float()[:, :, t0:t0 + T, y0:y0 + H, x0:x0 + W].contiguous()        # [1, 24, T, H, W]
    save_file({'samples.video': z}, a.out_latents)
    config = json.load(open(a.config))
    model = NonCausalViTDecoder(config['latents_mean'], config['latents_std'], KEEP[24], time_norm=5,
                                heads=config['decoder_num_attention_heads'], dim_head=config['decoder_attention_head_dim'],
                                rope_theta=config['decoder_rope_theta'], rope_dim_ratio=config['decoder_rope_dim_ratio'],
                                eps=config['decoder_norm_eps'], latent_channels=config['latent_channels'])
    model.load_state_dict(load_file(a.weights, device='cpu'), strict=True)
    model.eval().requires_grad_(False)
    st = time.time()
    with torch.no_grad():
        px = model(z.transpose(1, 2).float())                                         # [1, 3, F, 16H, 16W]
    mean = torch.tensor((.485, .456, .406)).view(1, 3, 1, 1, 1)
    std = torch.tensor((.229, .224, .225)).view(1, 3, 1, 1, 1)
    img = (px * std + mean).clamp(0, 1)[0].permute(1, 2, 3, 0).contiguous()          # [F, 16H, 16W, 3]
    save_file({'images': img}, a.out_dump)
    print(f'latents {tuple(z.shape)} -> frames {tuple(img.shape)} in {time.time() - st:.1f} s (CPU float32)')


if __name__ == '__main__':
    main()
