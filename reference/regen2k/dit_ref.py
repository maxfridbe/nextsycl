"""Open-H3-Regenerate-2K-IC's denoiser call, as the release runs it - Diffusers' MiniMaxH3Transformer3DModel with its
infer/ref2va_ic code: HyperFlow merged and its TwoTimeEmbedder, LMS Sharp merged with its 50 AdaLN curve branches, the
IC LoRA unmerged, the packed Ref2VA layout and per-row timesteps of layout.py - against the engine's (`regen2k` job
with `check_dump`): the same inputs (the draft's reference latent as the rows see it, x at the first noise level, the
silence, the text encoder's conditioning and tags), one call, the velocity compared.

The weights are the int8 ComfyUI checkpoint the engine runs, un-rotated to float (so the comparison is of the
implementations, not of the quantization), the full AdaLN projections and time embedder from ComfyUI's bf16 Ref2VA file
(fetch.py); CPU, float32, one block in memory at a time.

    python -I dit_ref.py --code <Open-H3-Regenerate-2K-IC>/infer --config <...>/base_model/transformer_ref/config.json \
        --int8 minimax_h3_ref2va_pruned_int8_convrot.safetensors --adaln minimax_h3_ref2va_bf16_adaln.safetensors \
        --ic open_h3_regenerate_2k_ic_lora.safetensors --hyperflow minimax_h3_hyperflow_8step_v1.0.safetensors \
        --lms minimax_h3_lms_v1.0_r64.safetensors --dump engine_check.safetensors [--blocks N]
"""
import argparse
import copy
import json
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import load_file

H = None


def hadamard(g=256):
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32)
    m = torch.ones(1, 1)
    while m.shape[0] < g:
        m = torch.kron(m, h4)
    return m / (g ** 0.5)


class Int8File:
    """The ComfyUI int8 ConvRot checkpoint: W = (q * scale) H per 256 group"""

    def __init__(self, path):
        self.f = safe_open(path, 'pt')
        self.keys = set(self.f.keys())

    def get(self, name):
        return self.f.get_tensor(name)

    def linear(self, name):
        w = self.f.get_tensor(name + '.weight')
        if w.dtype != torch.int8:
            return w.float()
        q = w.float() * self.f.get_tensor(name + '.weight_scale').float()
        n, k = q.shape
        return (q.view(n, k // 256, 256) @ H).view(n, k)


def comfy_block(ck, prefix):
    """A ComfyUI block's (or refiner block's) weights under Diffusers' names, float32"""
    qkv = ck.linear(f'{prefix}.attn.qkv_proj')
    q, k, v = qkv.chunk(3, 0)
    fc1 = ck.linear(f'{prefix}.mlp.fc1')
    gate, value = fc1.chunk(2, 0)
    sd = {'attn.to_q.weight': q, 'attn.to_k.weight': k, 'attn.to_v.weight': v,
          'attn.to_out.0.weight': ck.linear(f'{prefix}.attn.out_proj'),
          'ff.net.0.proj.weight': torch.cat([value, gate], 0), 'ff.net.2.weight': ck.linear(f'{prefix}.mlp.fc2'),
          'norm1.weight': ck.get(f'{prefix}.norm1.weight').float(), 'norm2.weight': ck.get(f'{prefix}.norm2.weight').float(),
          'attn.norm_q.weight': ck.get(f'{prefix}.attn.q_norm.weight').float(), 'attn.norm_k.weight': ck.get(f'{prefix}.attn.k_norm.weight').float()}
    return sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--code', required=True, help='the release\'s infer/ folder (ref2va_ic)')
    ap.add_argument('--config', required=True)
    ap.add_argument('--int8', required=True)
    ap.add_argument('--adaln', required=True)
    ap.add_argument('--ic', required=True)
    ap.add_argument('--hyperflow', required=True)
    ap.add_argument('--lms', required=True)
    ap.add_argument('--sharp', type=float, default=1.0)
    ap.add_argument('--dump', required=True)
    ap.add_argument('--threads', type=int, default=16)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    sys.path.insert(0, a.code)
    global H
    H = hadamard()
    from diffusers import MiniMaxH3Transformer3DModel
    from ref2va_ic import hyperflow, layout as lay, lora as ic_lib, sharp as sharp_lib

    cfg = json.load(open(a.config))
    cfg = {k: v for k, v in cfg.items() if not k.startswith('_')}
    with torch.device('meta'):
        model = MiniMaxH3Transformer3DModel(**cfg)
    ck = Int8File(a.int8)
    full = safe_open(a.adaln, 'pt')
    hf_sd = load_file(a.hyperflow)
    with safe_open(a.hyperflow, 'pt') as f:
        hf_meta = f.metadata()
    hf_scale = float(hf_meta['lora_alpha']) / float(hf_meta['lora_rank'])
    ic_sd = load_file(a.ic)
    lms_lin, lms_curves = sharp_lib.convert(a.lms)

    def put(module, sd):
        module.to_empty(device='cpu')
        missing, unexpected = module.load_state_dict(sd, strict=False)
        assert not unexpected, unexpected
        return missing

    # the small parts
    put(model.proj_in, {'weight': ck.get('video_patch_proj.weight').float(), 'bias': ck.get('video_patch_proj.bias').float()})
    put(model.audio_proj_in, {'weight': ck.get('audio_patch_proj.weight').float(), 'bias': ck.get('audio_patch_proj.bias').float()})
    put(model.context_embedder, {'weight': ck.get('condition_proj.weight').float(), 'bias': ck.get('condition_proj.bias').float()})
    put(model.time_embedder, {'linear_1.weight': full.get_tensor('time_embedder.proj_in.weight').float(), 'linear_1.bias': full.get_tensor('time_embedder.proj_in.bias').float(),
                              'linear_2.weight': full.get_tensor('time_embedder.proj_out.weight').float(), 'linear_2.bias': full.get_tensor('time_embedder.proj_out.bias').float()})
    put(model.norm_out, {'norm.weight': ck.get('final_layer.norm.weight').float(), 'linear.weight': full.get_tensor('final_layer.adaln_proj.linear.weight').float(),
                         'linear.bias': full.get_tensor('final_layer.adaln_proj.linear.bias').float()})
    put(model.proj_out, {'weight': ck.get('final_layer.video_out.weight').float(), 'bias': ck.get('final_layer.video_out.bias').float()})
    put(model.audio_proj_out, {'weight': ck.get('final_layer.audio_out.weight').float(), 'bias': ck.get('final_layer.audio_out.bias').float()})
    model.rope.to_empty(device='cpu')
    # the rope's buffers: recomputed as the class does (they are not in the checkpoint)
    with torch.device('cpu'):
        fresh = type(model.rope)(rope_freq_dim=cfg['rope_freq_dim'], rope_theta=cfg['rope_theta'])
    model.rope = fresh
    put(model.token_refiner.final_norm, {'weight': ck.get('token_refiner.final_norm.weight').float()})
    for i, blk in enumerate(model.token_refiner.refiner_blocks):
        put(blk, comfy_block(ck, f'token_refiner.blocks.{i}'))
    model.time_proj = model.time_proj.to('cpu')

    def merge(module_name, linear, sd, prefix, scale):
        a_, b_ = sd.get(f'{prefix}{module_name}.lora_A.weight'), sd.get(f'{prefix}{module_name}.lora_B.weight')
        if a_ is None:
            return 0
        with torch.no_grad():
            linear.weight += scale * (b_.float() @ a_.float())
        return 1

    # HyperFlow: the two-time embedder (its two LoRAs merged in float32), as hyperflow.install does
    te_base = model.time_embedder
    endpoint = copy.deepcopy(te_base)
    for name in ('linear_1', 'linear_2'):
        merge(f'time_embedder.{name}', getattr(te_base, name), hf_sd, 'transformer.', hf_scale)
        merge(f'endpoint_time_embedder.{name}', getattr(endpoint, name), hf_sd, 'transformer.', hf_scale)
    model.time_embedder = hyperflow.TwoTimeEmbedder(te_base, endpoint, copy.deepcopy(model.time_proj), float(hf_meta['hyperflow_gate']))
    # LMS Sharp's curve: the table's own coordinates, at the model's timestep (sharp.py's CurveContext)
    table = ck.get('adaln_t_table').float().contiguous()
    context = sharp_lib.CurveContext.__new__(sharp_lib.CurveContext)
    context.table, context.strength, context.coords = table, float(a.sharp), None
    model.register_forward_pre_hook(context.pre, with_kwargs=True)

    def lora_all(root, module):
        """HyperFlow (merged), LMS Sharp (merged), the IC LoRA (wrapped) on a block's linears: release order"""
        n = 0
        for name, lin in list(module.named_modules()):
            if not isinstance(lin, torch.nn.Linear):
                continue
            full_name = f'{root}.{name}'
            n += merge(full_name, lin, hf_sd, 'transformer.', hf_scale)
            if full_name in lms_lin and a.sharp:
                la, lb = lms_lin[full_name]
                with torch.no_grad():
                    lin.weight += a.sharp * (lb.float() @ la.float())
                n += 1
        for name, lin in list(module.named_modules()):
            key = f'transformer.{root}.{name}'
            if isinstance(lin, torch.nn.Linear) and key + '.lora_A.weight' in ic_sd:
                parent_name, child = name.rsplit('.', 1)
                parent = module.get_submodule(parent_name)
                wrapped = ic_lib.LoRALinear(lin, ic_sd[key + '.lora_A.weight'].float(), ic_sd[key + '.lora_B.weight'].float(), 1.0)
                if child.isdigit():
                    parent[int(child)] = wrapped
                else:
                    setattr(parent, child, wrapped)
                n += 1
        return n

    for i, blk in enumerate(model.token_refiner.refiner_blocks):
        lora_all(f'token_refiner.refiner_blocks.{i}', blk)

    # the blocks: weights in just before each runs, gone after
    t_load = [0.0]

    def pre(i):
        def hook(module, args):
            t0 = time.time()
            sd = comfy_block(ck, f'blocks.{i}')
            sd['adaln_proj.linear.weight'] = full.get_tensor(f'blocks.{i}.adaln_proj.linear.weight').float()
            sd['adaln_proj.linear.bias'] = full.get_tensor(f'blocks.{i}.adaln_proj.linear.bias').float()
            put(module, sd)
            lora_all(f'transformer_blocks.{i}', module)
            name = f'transformer_blocks.{i}.adaln_proj.linear'
            if a.sharp and name in lms_curves:
                ca, cb = lms_curves[name]
                module.adaln_proj.linear = sharp_lib.CurveLinear(module.adaln_proj.linear, ca.float(), cb.float(), context)
            t_load[0] += time.time() - t0
        return hook

    def post(module, args, out):
        module.to('meta')
        return out

    for i, blk in enumerate(model.transformer_blocks):
        blk.register_forward_pre_hook(pre(i))
        blk.register_forward_hook(post)
    model.eval()

    # the engine's inputs
    d = load_file(a.dump)
    anchor, x, audio = d['anchor'], d['x'], d['audio']                  # [24,T,h,w], [24,T,H,W], [32,2,A]
    sigmas = d['sigmas'].tolist()
    sigma, nxt = sigmas[0], sigmas[1]
    tags = d['tags'].long()
    prompt = {'prompt_embeds': d['context'].float(), 'text_token_tags': tags}
    target_shape = list(x.shape)
    ref, aud, emb, layout, n = lay.pack(anchor, target_shape, prompt, audio.permute(1, 0, 2).contiguous(), 'cpu')
    xr = lay.patch_rows(x)
    times, indices = lay.row_timesteps(layout, sigma)
    model.time_embedder.endpoint_timesteps = hyperflow.endpoints(times, lay.canonical_sigma(sigma), nxt)
    print(f'tokens {layout.token_tags.numel()}, timesteps {[round(float(t), 6) for t in times]}, endpoints '
          f'{[round(float(t), 6) for t in model.time_embedder.endpoint_timesteps]}', flush=True)
    t0 = time.time()
    with torch.inference_mode():
        out = model(hidden_states=torch.cat([ref.float(), xr.float()], 1), audio_hidden_states=aud.float(), encoder_hidden_states=emb.float(),
                    timestep=times, timestep_indices=indices, token_tags=layout.token_tags, position_ids=layout.position_ids,
                    video_indices=layout.video_indices, audio_indices=layout.audio_indices, text_indices=layout.text_indices,
                    attention_kwargs={}, return_dict=False)
    v = lay.unpatch_rows(out[0][:, n:].float(), target_shape)
    print(f'forward {time.time() - t0:.0f} s (weights {t_load[0]:.0f} s)', flush=True)
    ours = d['v'].float()

    def cmp(a_, b_):
        rel = float((a_ - b_).norm() / b_.norm())
        cos = float(torch.nn.functional.cosine_similarity(a_.double().flatten(), b_.double().flatten(), 0))
        return rel, cos
    for sign, name in ((1, 'v'), (-1, '-v')):
        rel, cos = cmp(sign * ours, v)
        print(f'engine {name} vs the release: rel {rel:.4f}, cosine {cos:.6f}')
    # per frame, the worst
    worst = min(cmp(-ours[:, t], v[:, t])[1] for t in range(v.shape[1]))
    print(f'worst frame cosine (engine -v): {worst:.6f}')
    # the denoised estimate x0 = x + sigma * v (the release's sign) - what a step moves toward
    x0_ref = x + sigma * v
    x0_ours = x - sigma * ours
    print('x0 at the first level: rel {:.4f}, cosine {:.6f}'.format(*cmp(x0_ours, x0_ref)))


if __name__ == '__main__':
    main()
