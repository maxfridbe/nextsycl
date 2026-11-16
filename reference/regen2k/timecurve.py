"""Open-H3-Regenerate-2K-IC's time conditioning in the curve form of the ComfyUI checkpoint our engine runs - the check
that the int8 Ref2VA file is the base the IC LoRA was trained on, and that HyperFlow's two-time embedding and LMS
Sharp's curve branches can be carried by the 8-wide timestep curve (`adaln_t_table`) instead of the full time embedder.

The curve form: ComfyUI's pruned checkpoint replaced `silu(time_embedder(t))` [2688] by 8 coordinates c(t) read from a
[1025, 8] table, and every AdaLN projection W [rows, 2688] by W U [rows, 8] (bias b + W m). A HyperFlow embedding
mix(t, r) = e_t(t) + g (e_r(r) - e_t(t)) is not on that curve; this script fits (m, U) from the base time embedder and
the table, projects silu(mix) on it, and measures what the projection loses in block 0's and the final layer's
modulation against the full projection of the Diffusers checkpoint (`base_small.safetensors`, fetch.py).

    python -I timecurve.py --base base_small.safetensors --int8 minimax_h3_ref2va_pruned_int8_convrot.safetensors \
        --hyperflow minimax_h3_hyperflow_8step_v1.0.safetensors --lms minimax_h3_lms_v1.0_r64.safetensors [--out dump.safetensors]

--out writes the modulation tables of the up3 / up4 / hf8 points (block 0 and the final layer, full form) for the
engine's check (`nextsycl video regen2k-check-time`).
"""
import argparse
import json

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SIGMAS_HF8 = [1.0, 0.99391, 0.984288, 0.966064, 0.923077, 0.834942, 0.696852, 0.468753, 0.0]


def tensors(path, names=None, prefix=None):
    out = {}
    with safe_open(path, 'pt') as f:
        for k in f.keys():
            if (names and k in names) or (prefix is not None and k.startswith(prefix)):
                out[k] = f.get_tensor(k)
        meta = f.metadata() or {}
    return out, meta


def sinusoid(t, dim=256):
    from diffusers.models.embeddings import get_timestep_embedding
    return get_timestep_embedding(t, dim, flip_sin_to_cos=True, downscale_freq_shift=0)


def embedder(w, t, lora=None, scale=1.0):
    """time_embedder(time_proj(t)) in float32: linear_1, silu, linear_2 (with a LoRA on both when given)"""
    def lin(name, x):
        W = w[f'time_embedder.{name}.weight'].float()
        if lora is not None:
            W = W + scale * lora[name][1].float() @ lora[name][0].float()
        return x @ W.T + w[f'time_embedder.{name}.bias'].float()
    x = sinusoid(t)
    return lin('linear_2', torch.nn.functional.silu(lin('linear_1', x)))


def rel(a, b):
    return float((a - b).norm() / b.norm())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--int8', required=True)
    ap.add_argument('--hyperflow', required=True)
    ap.add_argument('--lms', required=True)
    ap.add_argument('--out')
    a = ap.parse_args()
    base, _ = tensors(a.base, prefix='')
    ck, _ = tensors(a.int8, names={'adaln_t_table', 'blocks.0.adaln_proj.linear.weight', 'blocks.0.adaln_proj.linear.bias',
                                   'final_layer.adaln_proj.linear.weight', 'final_layer.adaln_proj.linear.bias',
                                   'blocks.0.attn.qkv_proj.weight', 'blocks.0.attn.qkv_proj.weight_scale'})
    table = ck['adaln_t_table'].float()
    grid = table.shape[0]

    # 1. the int8 file is the base: block 0's q matrix, un-rotated (ConvRot: W_rot = W H per 256 group, H symmetric)
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32)
    H = torch.ones(1, 1)
    while H.shape[0] < 256:
        H = torch.kron(H, h4)
    H = H / 16.0
    q = ck['blocks.0.attn.qkv_proj.weight'][:7168].float() * ck['blocks.0.attn.qkv_proj.weight_scale'][:7168].float()
    w = (q.view(7168, -1, 256) @ H).view(7168, -1)
    ref = base['transformer_blocks.0.attn.to_q.weight'].float()
    print(f'base match: int8 q (un-rotated) vs Diffusers to_q: rel {rel(w, ref):.5f}, cos {float(torch.nn.functional.cosine_similarity(w.flatten(), ref.flatten(), 0)):.6f}')

    # 2. fit silu(e(t_k)) = m + U c_k over the table's grid
    tk = torch.linspace(0, 1, grid, dtype=torch.float64).float()
    S = torch.nn.functional.silu(embedder(base, tk)).double()                      # [grid, 2688]
    X = torch.cat([torch.ones(grid, 1, dtype=torch.float64), table.double()], 1)   # [grid, 9]
    theta = torch.linalg.lstsq(X, S).solution                                      # [9, 2688]
    fit = X @ theta
    print(f'curve fit: silu(e(t)) = m + U c(t), rel residual {rel(fit, S):.2e}')
    m, U = theta[0], theta[1:].T                                                   # [2688], [2688, 8]
    pinv = torch.linalg.pinv(U)                                                    # [8, 2688]

    # 3. the curve form against the full form, block 0 and the final layer, on the grid
    def full(name, s):
        return s @ base[f'{name}.weight'].double().T + base[f'{name}.bias'].double()

    def pruned(name, c):
        return c @ ck[f'{name}.weight'].double().T + ck[f'{name}.bias'].double()
    for full_name, ck_name in (('transformer_blocks.0.adaln_proj.linear', 'blocks.0.adaln_proj.linear'),
                               ('norm_out.linear', 'final_layer.adaln_proj.linear')):
        print(f'{ck_name}: curve form vs full on the grid: rel {rel(pruned(ck_name, table.double()), full(full_name, S)):.2e}')

    # 4. HyperFlow: the two-time embedding at the IC sampler's points, projected on the curve
    hf, meta = tensors(a.hyperflow, prefix='transformer.')
    gate = float(meta['hyperflow_gate'])
    scale = float(meta['lora_alpha']) / float(meta['lora_rank'])
    lt = {n: (hf[f'transformer.time_embedder.{n}.lora_A.weight'], hf[f'transformer.time_embedder.{n}.lora_B.weight']) for n in ('linear_1', 'linear_2')}
    le = {n: (hf[f'transformer.endpoint_time_embedder.{n}.lora_A.weight'], hf[f'transformer.endpoint_time_embedder.{n}.lora_B.weight']) for n in ('linear_1', 'linear_2')}
    raw = json.loads(meta['hyperflow_sigmas'])
    sh = float(meta['hyperflow_video_shift'])
    grid9 = [sh * s / (1 + (sh - 1) * s) for s in raw]
    print('HyperFlow grid (shift 12):', [round(s, 6) for s in grid9], 'gate', gate, 'scale', scale)
    points = []
    for s, n in zip(grid9[:-1], grid9[1:]):
        points.append((1 - s, 1 - n))                    # target (and text) rows: r = 1 - next sigma
    points += [(0.999, 0.999), (1.0, 1.0)]               # reference rows; the pinned silence
    t = torch.tensor([p[0] for p in points]).float()
    r = torch.tensor([p[1] for p in points]).float()
    et = embedder(base, t, lt, scale)
    er = embedder(base, r, le, scale)
    s_hf = torch.nn.functional.silu(et + gate * (er - et)).double()
    c_hf = (s_hf - m) @ pinv.T
    back = m + c_hf @ U.T
    out = {'points': torch.tensor(points, dtype=torch.float64), 'coords': c_hf}
    for i, (tt, rr) in enumerate(points):
        line = [f't {tt:.6f} r {rr:.6f}: embedding off the curve {rel(back[i], s_hf[i]):.2e}']
        for full_name, ck_name in (('transformer_blocks.0.adaln_proj.linear', 'blocks.0.adaln_proj.linear'),
                                   ('norm_out.linear', 'final_layer.adaln_proj.linear')):
            f = full(full_name, s_hf[i:i + 1])
            p = pruned(ck_name, c_hf[i:i + 1])
            line.append(f'{ck_name.split(".")[0]} modulation rel {rel(p, f):.2e}')
            out[f'{ck_name}.{i}'] = f[0]
        print('  ' + ', '.join(line))
    # without HyperFlow (r ignored), the table's own point, for scale
    plain = torch.nn.functional.silu(embedder(base, t)).double()
    print('  HyperFlow vs plain embedding at those points: rel', [round(rel(s_hf[i], plain[i]), 4) for i in range(len(points))])

    # 5. LMS Sharp: its AdaLN branches read the curve (A: [r, 8])
    lms, _ = tensors(a.lms, prefix='')
    curves = sorted({k.rsplit('.lora_', 1)[0] for k in lms if '.adaln_proj.linear' in k})
    shapes = {k: tuple(lms[k + '.lora_A.weight'].shape) for k in curves}
    print(f'LMS: {len(lms) // 2} LoRA pairs, {len(curves)} on AdaLN projections, A shapes {set(shapes.values())}; '
          f'final layer: {[k for k in lms if k.startswith("diffusion_model.final_layer")][:4]}')
    if a.out:
        save_file({k: v.float().contiguous() for k, v in out.items()}, a.out)


if __name__ == '__main__':
    main()
