"""How far the Ref2VA denoiser is from the FL2VA one the pi-H3 adapters were trained on, in the parts an adapter touches,
against how far the adapter itself moves them - the question for the adapter over Ref2VA (docs/ports/pi-h3.md).

For each part, relative distances ||A - B|| / ||B||: Ref2VA against FL2VA, the adapter against FL2VA (a replaced
tensor) or the adapter's LoRA update ||B A|| against FL2VA's matrix. Both bases int8 ConvRot, un-rotated (W = (q s) H).

    python -I ref2va_diff.py --fl2va FL2VA.safetensors --ref2va REF2VA.safetensors --adapter pi_h3_2step_dwd_adapter_int8_convrot.safetensors
"""
import argparse

import torch
from safetensors import safe_open


def hadamard(g=256):
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32)
    m = torch.ones(1, 1)
    while m.shape[0] < g:
        m = torch.kron(m, h4)
    return m / (g ** 0.5)


H = hadamard()


class F:
    def __init__(self, p):
        self.f = safe_open(p, 'pt')
        self.k = set(self.f.keys())

    def get(self, n):
        return self.f.get_tensor(n).float()

    def lin(self, n):
        w = self.f.get_tensor(n + '.weight')
        if w.dtype != torch.int8:
            return w.float()
        q = w.float() * self.f.get_tensor(n + '.weight_scale').float()
        r, c = q.shape
        return (q.view(r, c // 256, 256) @ H).view(r, c)


def rel(a, b):
    return float((a - b).norm() / b.norm())


def main():
    ap = argparse.ArgumentParser()
    for k in ('fl2va', 'ref2va', 'adapter'):
        ap.add_argument('--' + k, required=True)
    a = ap.parse_args()
    fl, rf, ad = F(a.fl2va), F(a.ref2va), F(a.adapter)
    print('| part | Ref2VA vs FL2VA | adapter vs FL2VA | adapter vs Ref2VA |')
    print('|---|---|---|---|')
    # AdaLN projections (every block: the adapter's replace them)
    rr, aa, ar = [], [], []
    for i in range(50):
        n = f'blocks.{i}.adaln_proj.linear.weight'
        f, r, x = fl.get(n), rf.get(n), ad.get(n)
        rr.append(rel(r, f)); aa.append(rel(x, f)); ar.append(rel(x, r))
    # the projections are on each checkpoint's own 8-coordinate curve (adaln_t_table): what they give - the modulation at a
    # timestep, W c(t) + b - is what compares
    tf, tr = fl.get('adaln_t_table'), rf.get('adaln_t_table')
    print(f'| adaln_t_table (the curve each projection is on) | {rel(tr, tf):.3f} | - (the adapter has none: FL2VA\'s) | - |')

    def curve(tab, t):
        pos = t * (tab.shape[0] - 1)
        i = min(int(pos), tab.shape[0] - 2)
        return tab[i] + (pos - i) * (tab[i + 1] - tab[i])
    ts = (0.0067, 0.1346, 0.5, 0.9, 0.999)
    mr, ma, mx = [], [], []
    for i in range(50):
        n = f'blocks.{i}.adaln_proj.linear'
        wf, wr, wa = fl.get(n + '.weight'), rf.get(n + '.weight'), ad.get(n + '.weight')
        bf, br = fl.get(n + '.bias'), rf.get(n + '.bias')
        ba = ad.get(n + '.bias') if n + '.bias' in ad.k else bf
        for t in ts:
            mf, mrf = wf @ curve(tf, t) + bf, wr @ curve(tr, t) + br
            mr.append(rel(mrf, mf))
            ma.append(rel(wa @ curve(tf, t) + ba, mf))
            mx.append(rel(wa @ curve(tr, t) + ba, mrf))

    def q(v):
        return f'{sorted(v)[len(v) // 2]:.3f} / {max(v):.3f}'
    print(f'| AdaLN modulation W c(t) + b, 50 blocks x t {ts} (median / max) | {q(mr)} | {q(ma)} | {q(mx)} (the adapter\'s projection on Ref2VA\'s curve) |',
          flush=True)
    print(f'| AdaLN projections, 50 blocks (median / max) | {sorted(rr)[25]:.3f} / {max(rr):.3f} | {sorted(aa)[25]:.3f} / {max(aa):.3f} | {sorted(ar)[25]:.3f} / {max(ar):.3f} |', flush=True)
    rb = [rel(rf.get(f'blocks.{i}.adaln_proj.linear.bias'), fl.get(f'blocks.{i}.adaln_proj.linear.bias')) for i in range(50)]
    print(f'| AdaLN biases (median / max; the adapter keeps the base\'s but on 46-49) | {sorted(rb)[25]:.3f} / {max(rb):.3f} | - | - |', flush=True)
    # the blocks the adapter replaces whole
    for i in (46, 47, 48, 49):
        for l in ('attn.out_proj', 'mlp.fc1', 'mlp.fc2'):
            n = f'blocks.{i}.{l}'
            f, r, x = fl.lin(n), rf.lin(n), ad.lin(n)
            print(f'| block {i} {l} (replaced) | {rel(r, f):.3f} | {rel(x, f):.3f} | {rel(x, r):.3f} |', flush=True)
        f, r = fl.lin(f'blocks.{i}.attn.qkv_proj'), rf.lin(f'blocks.{i}.attn.qkv_proj')
        print(f'| block {i} attn.qkv_proj (the base\'s) | {rel(r, f):.3f} | - | - |', flush=True)
    # the LoRA blocks: how far Ref2VA's matrix is from FL2VA's, against the LoRA's update
    for i in (0, 10, 20, 30, 40, 45):
        for l in ('attn.qkv_proj', 'attn.out_proj', 'mlp.fc1', 'mlp.fc2'):
            n = f'blocks.{i}.{l}'
            f, r = fl.lin(n), rf.lin(n)
            up = ''
            if n + '.lora_A.weight' in ad.k:
                d = ad.get(n + '.lora_B.weight') @ ad.get(n + '.lora_A.weight')
                up = f'{float(d.norm() / f.norm()):.3f} (LoRA update)'
            print(f'| block {i} {l} | {rel(r, f):.3f} | {up or "-"} | - |', flush=True)
    for n in ('final_layer.adaln_proj.linear.weight', 'final_layer.adaln_proj.linear.bias', 'final_layer.norm.weight', 'final_layer.video_out.weight',
              'final_layer.audio_out.weight'):
        f, r, x = fl.get(n), rf.get(n), ad.get(n)
        print(f'| {n} (replaced) | {rel(r, f):.3f} | {rel(x, f):.3f} | {rel(x, r):.3f} |', flush=True)


if __name__ == '__main__':
    main()
