"""HyperFlow's two-time modulation tables, as the release makes them (infer/ref2va_ic/hyperflow.py's TwoTimeEmbedder on
Diffusers' time embedder, the full AdaLN projection, LMS Sharp's curve branch on the table at t), against the ones the
engine made for the same (t, r) points (`hf.block0`, `hf.block49`, `hf.final` in a regen2k check dump).

    python -I time_ref.py --code <release>/infer --int8 <int8 ckpt> --adaln minimax_h3_ref2va_bf16_adaln.safetensors \
        --hyperflow minimax_h3_hyperflow_8step_v1.0.safetensors --lms minimax_h3_lms_v1.0_r64.safetensors --dump check.safetensors
"""
import argparse
import copy
import sys

import torch
from safetensors import safe_open
from safetensors.torch import load_file


def main():
    ap = argparse.ArgumentParser()
    for k in ('code', 'int8', 'adaln', 'hyperflow', 'lms', 'dump'):
        ap.add_argument('--' + k, required=True)
    ap.add_argument('--sharp', type=float, default=1.0)
    a = ap.parse_args()
    sys.path.insert(0, a.code)
    from diffusers.models.embeddings import TimestepEmbedding, Timesteps
    from ref2va_ic import hyperflow, sharp as sharp_lib

    full = safe_open(a.adaln, 'pt')
    te = TimestepEmbedding(in_channels=256, time_embed_dim=5376, out_dim=2688)
    te.load_state_dict({'linear_1.weight': full.get_tensor('time_embedder.proj_in.weight'), 'linear_1.bias': full.get_tensor('time_embedder.proj_in.bias'),
                        'linear_2.weight': full.get_tensor('time_embedder.proj_out.weight'), 'linear_2.bias': full.get_tensor('time_embedder.proj_out.bias')})
    hf = load_file(a.hyperflow)
    with safe_open(a.hyperflow, 'pt') as f:
        meta = f.metadata()
    scale = float(meta['lora_alpha']) / float(meta['lora_rank'])
    endpoint = copy.deepcopy(te)
    with torch.no_grad():
        for mod, key in ((te, 'time_embedder'), (endpoint, 'endpoint_time_embedder')):
            for n in ('linear_1', 'linear_2'):
                lin = getattr(mod, n)
                lin.weight += scale * hf[f'transformer.{key}.{n}.lora_B.weight'].float() @ hf[f'transformer.{key}.{n}.lora_A.weight'].float()
    proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
    two = hyperflow.TwoTimeEmbedder(te, endpoint, copy.deepcopy(proj), float(meta['hyperflow_gate']))
    d = load_file(a.dump)
    pts = d['hf.points'].double()
    t = pts[:, 0].float()
    two.endpoint_timesteps = pts[:, 1].float()
    with torch.no_grad():
        s = torch.nn.functional.silu(two(proj(t)))
    ck = safe_open(a.int8, 'pt')
    table = ck.get_tensor('adaln_t_table').float()
    pos = t.clamp(0, 1) * (len(table) - 1)
    lo = pos.floor().long().clamp(max=len(table) - 2)
    coords = torch.lerp(table[lo], table[lo + 1], (pos - lo).unsqueeze(-1))
    _, curves = sharp_lib.convert(a.lms)
    nb = 1 + max(int(k.split('.')[1]) for k in full.keys() if k.startswith('blocks.'))

    def cmp(name, ours, ref):
        rel = float((ours - ref).norm() / ref.norm())
        worst = max(float((ours[i] - ref[i]).norm() / ref[i].norm()) for i in range(len(ref)))
        print(f'{name}: rel {rel:.2e}, worst point {worst:.2e}')
    for key, block in (('hf.block0', 0), ('hf.block49', nb - 1)):
        w, b = full.get_tensor(f'blocks.{block}.adaln_proj.linear.weight').float(), full.get_tensor(f'blocks.{block}.adaln_proj.linear.bias').float()
        ref = s @ w.T + b
        ca, cb = curves[f'transformer_blocks.{block}.adaln_proj.linear']
        ref = ref + a.sharp * (coords @ ca.float().T) @ cb.float().T
        cmp(key, d[key].float(), ref)
    w, b = full.get_tensor('final_layer.adaln_proj.linear.weight').float(), full.get_tensor('final_layer.adaln_proj.linear.bias').float()
    cmp('hf.final', d['hf.final'].float(), s @ w.T + b)
    print('points (t, r):', [(round(float(x), 6), round(float(y), 6)) for x, y in pts])


if __name__ == '__main__':
    main()
