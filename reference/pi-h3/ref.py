"""pi-H3 (Lakonik/pi-H3) as LakonLab runs it: its MiniMaxH3Transformer3DModel (the pruned H3 with the adapter: the
block LoRAs merged, the adapter's AdaLN projections, its four fully tuned last blocks and final layer, and with DWD its
`dwd_norm_out` and Sol pixel head), its FlowEulerODEScheduler for the video (shift 12) and the audio (shift 3), its
pipeline's sampling loop (pipeline_minimax_h3.py: pure noise in, `t = 1 - sigma`, Euler, no CFG) and DWD's last step -
on the CPU in float32, writing a run dump the engine's `denoise` job replays (`nextsycl video job denoise`): the noise,
the text conditioning, the schedule, every step's denoised estimate, the finished latents and, with DWD, the last step's
final hidden states of the video rows and the head's pixels.

LakonLab is read from a checkout (--lakonlab, pinned at 2ee2b334): its own source for the transformer, the DWD head
(dwd_head.py, sol_h3/), the scheduler and the SwiGLU op. Its package __init__ files import its whole training stack
(mmcv 1.6, triton, ...); here they are left out and only these leaf modules are loaded, with mmcv's three init helpers
and its registry (all LakonLab's builder needs) as small stand-ins.

The weights are the ones the engine runs - the int8 ConvRot base and adapter un-rotated to float (W = (q s) H per 256
group) - so the comparison is of the implementations, not of the quantization; `--adapter` may also be a Diffusers
adapter folder's file (the VAE adapters, or the DWD adapter in fp16 to see what int8 costs). The text conditioning is
the engine's own (its `encode` job; the text encoder is H3's, unchanged by the adapters), one block of the denoiser in
memory at a time.

    python -I ref.py --lakonlab LakonLab --base minimax_h3_fl2va_pruned_int8_convrot.safetensors \
        --adapter pi_h3_2step_dwd_adapter_int8_convrot.safetensors --config pi_h3_2step_dwd_adapter/config.json \
        --context ctx.safetensors --size 448x256 --frames 22 --steps 2 --seed 42 --out dump.safetensors
"""
import argparse
import json
import logging
import sys
import time
import types
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

MAX_RAW_T = {2: 0.93, 4: 0.96}
FINAL_STEP_SIZE_SCALE = 0.6
# Sol's ImageNet-normalized RGB -> [-1, 1] (LakonLab's RGBColorEncoder in the pi-H3 example)
COLOR_MEAN = (-0.03, -0.088, -0.188)
COLOR_STD = (0.458, 0.448, 0.45)


def load_lakonlab(root):
    """LakonLab's leaf modules, without its package __init__ files (they import the training stack)"""
    root = Path(root) / 'lakonlab'
    nn = torch.nn

    def constant_init(module, val, bias=0):
        if getattr(module, 'weight', None) is not None:
            nn.init.constant_(module.weight, val)
        if getattr(module, 'bias', None) is not None:
            nn.init.constant_(module.bias, bias)

    def xavier_init(module, gain=1, bias=0, distribution='normal'):
        (nn.init.xavier_uniform_ if distribution == 'uniform' else nn.init.xavier_normal_)(module.weight, gain=gain)
        if getattr(module, 'bias', None) is not None:
            nn.init.constant_(module.bias, bias)

    def kaiming_init(module, a=0, mode='fan_out', nonlinearity='relu', bias=0, distribution='normal'):
        (nn.init.kaiming_uniform_ if distribution == 'uniform' else nn.init.kaiming_normal_)(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        if getattr(module, 'bias', None) is not None:
            nn.init.constant_(module.bias, bias)

    class Registry:
        def __init__(self, name):
            self.name, self.module_dict = name, {}

        def get(self, key):
            return self.module_dict.get(key)

        def register_module(self, name=None, force=False, module=None):
            def add(cls):
                self.module_dict[name or cls.__name__] = cls
                return cls
            if module is not None:
                return add(module)
            if isinstance(name, type):
                cls, name = name, None
                return add(cls)
            return add

    def build_from_cfg(cfg, registry, default_args=None):
        args = dict(cfg)
        kind = args.pop('type')
        cls = registry.get(kind) if isinstance(kind, str) else kind
        if cls is None:
            raise KeyError(f'{kind} is not in the {registry.name} registry')
        for k, v in (default_args or {}).items():
            args.setdefault(k, v)
        return cls(**args)

    mmcv, cnn, utils = (types.ModuleType(n) for n in ('mmcv', 'mmcv.cnn', 'mmcv.utils'))
    cnn.constant_init, cnn.xavier_init, cnn.kaiming_init = constant_init, xavier_init, kaiming_init
    utils.Registry, utils.build_from_cfg = Registry, build_from_cfg
    mmcv.cnn, mmcv.utils = cnn, utils
    sys.modules.update({'mmcv': mmcv, 'mmcv.cnn': cnn, 'mmcv.utils': utils})
    for name in ('lakonlab', 'lakonlab.models', 'lakonlab.models.architectures', 'lakonlab.models.architectures.diffusers',
                 'lakonlab.models.architectures.sol_h3', 'lakonlab.models.diffusions', 'lakonlab.models.diffusions.schedulers',
                 'lakonlab.ops', 'lakonlab.runner', 'lakonlab.utils'):
        m = types.ModuleType(name)
        m.__path__ = [str(root.joinpath(*name.split('.')[1:]))]
        sys.modules[name] = m
    lu = sys.modules['lakonlab.utils']
    lu.get_root_logger = lambda *a, **k: logging.getLogger('lakonlab')

    def rgetattr(obj, attr, *args):
        for a in attr.split('.'):
            obj = getattr(obj, a, *args)
        return obj
    lu.rgetattr = rgetattr
    att = types.ModuleType('lakonlab.ops.attention')
    att.flash_attn_interface = att.sageattn = att.sol_attention = None
    sys.modules['lakonlab.ops.attention'] = att
    ck = types.ModuleType('lakonlab.runner.checkpoint')

    def unused(*a, **k):
        raise RuntimeError('LakonLab checkpoint loading is not used by this reference')
    ck._load_cached_checkpoint = ck.load_full_state_dict = unused
    sys.modules['lakonlab.runner.checkpoint'] = ck
    import lakonlab.ops.activation as act
    ops = sys.modules['lakonlab.ops']
    ops.swiglu_linear, ops.silu_mul_, ops.attention = act.swiglu_linear, act.silu_mul_, att
    ops.quantize_frozen_linears = ops.InplaceLoraLinear = None
    import lakonlab.models.architectures.sol_h3.sol_h3  # noqa: F401  (registers SolH3DWDHead)
    from lakonlab.models.architectures.diffusers.minimax_h3 import MiniMaxH3Transformer3DModel
    from lakonlab.models.diffusions.schedulers.flow_euler_ode import FlowEulerODEScheduler
    return MiniMaxH3Transformer3DModel, FlowEulerODEScheduler


def hadamard(g=256):
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32)
    m = torch.ones(1, 1)
    while m.shape[0] < g:
        m = torch.kron(m, h4)
    return m / (g ** 0.5)


H = None


def swap(w):
    """ComfyUI's mlp.fc1 holds the two SwiGLU halves of Diffusers' ff.net.0.proj in the other order"""
    a, b = w.chunk(2, 0)
    return torch.cat([b, a], 0)


class Weights:
    """A safetensors file under ComfyUI's names, float32 on request: int8 ConvRot layers un-rotated; a Diffusers-layout
    adapter's names (transformer_blocks.N.attn.to_out.0 ...) mapped to ComfyUI's, fc1's halves swapped to its order"""

    BLOCK = (('.attn.to_out.0.', '.attn.out_proj.'), ('.ff.net.0.proj.', '.mlp.fc1.'), ('.ff.net.2.', '.mlp.fc2.'))
    TOP = {'norm_out.linear.': 'final_layer.adaln_proj.linear.', 'norm_out.norm.': 'final_layer.norm.', 'proj_out.': 'final_layer.video_out.',
           'audio_proj_out.': 'final_layer.audio_out.'}

    def __init__(self, path):
        self.f = safe_open(path, 'pt')
        self.names = {}
        for k in self.f.keys():
            c = k
            if c.startswith('transformer_blocks.'):
                c = 'blocks.' + c[len('transformer_blocks.'):]
                for a, b in self.BLOCK:
                    c = c.replace(a, b)
            for a, b in self.TOP.items():
                if c.startswith(a):
                    c = b + c[len(a):]
            self.names[c] = k
        self.diffusers = any(k.startswith('transformer_blocks.') for k in self.f.keys())

    def __contains__(self, name):
        return name in self.names

    def get(self, name):
        t = self.f.get_tensor(self.names[name]).float()
        if self.diffusers and '.mlp.fc1.' in name and not name.endswith('lora_A.weight'):
            t = swap(t)
        return t

    def linear(self, name):
        """name.weight as float: un-rotated when int8 ConvRot"""
        w = self.f.get_tensor(self.names[name + '.weight'])
        if w.dtype != torch.int8:
            return self.get(name + '.weight')
        q = w.float() * self.f.get_tensor(self.names[name + '.weight_scale']).float()
        n, k = q.shape
        return (q.view(n, k // 256, 256) @ H).view(n, k)


def main():
    ap = argparse.ArgumentParser()
    for k in ('lakonlab', 'base', 'adapter', 'config', 'context', 'out'):
        ap.add_argument('--' + k, required=True)
    ap.add_argument('--size', default='448x256', help='WxH, multiples of 32')
    ap.add_argument('--frames', type=int, default=22, help='17n + 5')
    ap.add_argument('--steps', type=int, default=2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--threads', type=int, default=16)
    ap.add_argument('--perturb', type=float, default=0.0,
                    help='relative noise added to the stream after block 0 at every step (a sensitivity check: what an error of that size there becomes)')
    ap.add_argument('--taps', default='', help='blocks whose output (the video rows) the first step dumps as tap.N: 10,46,49')
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    global H
    H = hadamard()
    Model, Scheduler = load_lakonlab(a.lakonlab)
    cfg = {k: v for k, v in json.load(open(a.config)).items() if not k.startswith('_')}
    dwd = cfg.get('dwd_head_config') is not None
    t0 = time.time()
    model = Model(**cfg, checkpointing=False, torch_dtype='float32')
    model.requires_grad_(False).eval()
    base, ad = Weights(a.base), Weights(a.adapter)
    print(f'adapter: {a.adapter} ({"Diffusers" if ad.diffusers else "ComfyUI"} layout, {len(ad.names)} tensors), DWD {dwd}', flush=True)

    def put(module, sd):
        module.to_empty(device='cpu')
        missing, unexpected = module.load_state_dict(sd, strict=False)
        assert not unexpected, unexpected
        missing = [m for m in missing if not m.startswith('rope.')]
        assert not missing, missing

    def pick(name):
        """the adapter's tensor where it has one, else the base's"""
        return ad.get(name) if name in ad else base.get(name)

    put(model.proj_in, {'weight': base.get('video_patch_proj.weight'), 'bias': base.get('video_patch_proj.bias')})
    put(model.audio_proj_in, {'weight': base.get('audio_patch_proj.weight'), 'bias': base.get('audio_patch_proj.bias')})
    put(model.context_embedder, {'weight': base.get('condition_proj.weight'), 'bias': base.get('condition_proj.bias')})
    model.time_embedder.to_empty(device='cpu')
    model.time_embedder.table.copy_(base.get('adaln_t_table'))
    put(model.norm_out, {'norm.weight': pick('final_layer.norm.weight'), 'linear.weight': pick('final_layer.adaln_proj.linear.weight'),
                         'linear.bias': pick('final_layer.adaln_proj.linear.bias')})
    put(model.proj_out, {'weight': pick('final_layer.video_out.weight'), 'bias': pick('final_layer.video_out.bias')})
    put(model.audio_proj_out, {'weight': pick('final_layer.audio_out.weight'), 'bias': pick('final_layer.audio_out.bias')})
    if dwd:
        put(model.dwd_norm_out, {'linear.weight': ad.get('dwd_norm_out.linear.weight'), 'linear.bias': ad.get('dwd_norm_out.linear.bias')})
        ref = model.dwd_pixel_refiner
        sd = {}
        for k in ref.state_dict():
            name = 'dwd_pixel_refiner.' + k
            if k.endswith('.weight') and name[:-len('.weight')] + '.weight_scale' in ad:
                sd[k] = ad.linear(name[:-len('.weight')])
            else:
                sd[k] = ad.get(name)
        put(ref, sd)
    # rope: its buffers as the class makes them (not in any checkpoint)
    for n, b in list(model.rope.named_buffers()):
        assert b.device.type == 'cpu', f'rope.{n} on {b.device}'

    def block_sd(prefix, i, refiner=False):
        """a block under Diffusers' names: the base's matrices (the adapter's where it replaces them) with the adapter's
        LoRAs merged in float32, as load_lakonlab_adapter's fused update"""
        p = f'{prefix}.{i}'
        lin = lambda n: ad.linear(f'{p}.{n}') if f'{p}.{n}.weight' in ad else base.linear(f'{p}.{n}')  # noqa: E731
        q, k, v = base.linear(f'{p}.attn.qkv_proj').chunk(3, 0)
        mats = {'attn.out_proj': lin('attn.out_proj'), 'mlp.fc1': lin('mlp.fc1'), 'mlp.fc2': lin('mlp.fc2')}
        for n, w in mats.items():
            la, lb = f'{p}.{n}.lora_A.weight', f'{p}.{n}.lora_B.weight'
            if la in ad:
                w += ad.get(lb) @ ad.get(la)
        sd = {'attn.to_q.weight': q, 'attn.to_k.weight': k, 'attn.to_v.weight': v, 'attn.to_out.0.weight': mats['attn.out_proj'],
              'ff.net.0.proj.weight': swap(mats['mlp.fc1']), 'ff.net.2.weight': mats['mlp.fc2'],
              'norm1.weight': base.get(f'{p}.norm1.weight'), 'norm2.weight': base.get(f'{p}.norm2.weight'),
              'attn.norm_q.weight': base.get(f'{p}.attn.q_norm.weight'), 'attn.norm_k.weight': base.get(f'{p}.attn.k_norm.weight')}
        if not refiner:
            sd['adaln_proj.linear.weight'] = pick(f'{p}.adaln_proj.linear.weight')
            sd['adaln_proj.linear.bias'] = pick(f'{p}.adaln_proj.linear.bias')
        return sd

    put(model.token_refiner.final_norm, {'weight': base.get('token_refiner.final_norm.weight')})
    for i, blk in enumerate(model.token_refiner.refiner_blocks):
        put(blk, block_sd('token_refiner.blocks', i, refiner=True))
    full = sorted({int(k.split('.')[1]) for k in ad.names if k.startswith('blocks.') and k.endswith('mlp.fc1.weight')})
    lora = sum(1 for k in ad.names if k.endswith('lora_A.weight'))
    print(f'adapter: {lora} LoRA pairs, blocks fully replaced {full}, AdaLN projections of '
          f'{sum(1 for k in ad.names if k.endswith("adaln_proj.linear.weight") and k.startswith("blocks."))} blocks', flush=True)
    t_load = [0.0]

    def pre(i):
        def hook(module, args):
            s = time.time()
            put(module, block_sd('blocks', i))
            t_load[0] += time.time() - s
        return hook

    taps = {int(b) for b in a.taps.split(',') if b.strip()}
    tapped, step = {}, [0]

    def post(i):
        def hook(module, args, out):
            module.to('meta')
            if i == 0 and a.perturb:
                g = torch.Generator().manual_seed(1234 + step[0])
                out = out + a.perturb * out.float().pow(2).mean().sqrt() * torch.randn(out.shape, generator=g).to(out.dtype)
            if step[0] == 0 and i in taps:
                tapped[f'tap.{i}'] = out[0, video_rows[0]].float().clone()
            return out
        return hook

    video_rows = [None]
    for i, blk in enumerate(model.transformer_blocks):
        blk.register_forward_pre_hook(pre(i))
        blk.register_forward_hook(post(i))
    print(f'model built in {time.time() - t0:.0f} s', flush=True)

    # the inputs: the engine's text conditioning, the engine's noise convention (torch.manual_seed, video then audio)
    ctx = load_file(a.context)['context'].float()                                     # [1, L, 5120]
    w, h = (int(x) for x in a.size.split('x'))
    assert w % 32 == 0 and h % 32 == 0 and a.frames % 17 == 5
    lt = (a.frames - 5) // 17 * 5 + 2
    at = round(a.frames / 24 * 40)
    torch.manual_seed(a.seed)
    noise_v = torch.randn(1, 24, lt, h // 16, w // 16)
    noise_a = torch.randn(1, 32, 2, at)
    sched = Scheduler(num_train_timesteps=1, shift=12.0, max_raw_t=MAX_RAW_T[a.steps], final_step_size_scale=FINAL_STEP_SIZE_SCALE)
    asched = Scheduler(num_train_timesteps=1, shift=3.0, max_raw_t=MAX_RAW_T[a.steps], final_step_size_scale=FINAL_STEP_SIZE_SCALE)
    sched.set_timesteps(a.steps)
    asched.set_timesteps(a.steps)
    print(f'video sigmas {[round(float(s), 6) for s in sched.sigmas]}, audio {[round(float(s), 6) for s in asched.sigmas]}', flush=True)
    print(f'clip: {w}x{h}, {a.frames} frames = {lt} latent frames, {at} audio latents; {ctx.shape[1]} text tokens', flush=True)

    # DWD: the final hidden states going into dwd_norm_out (all rows) and what it gives
    caught = {}
    if dwd:
        model.dwd_norm_out.register_forward_hook(lambda m, args, out: caught.update(h=args[0].detach().clone(), temb=args[1].detach().clone(),
                                                                                       idx=args[2].detach().clone(), features=out.detach().clone()))
    layout = model.prepare_layout(noise_v, noise_a, ctx.shape[1])
    video_rows[0] = layout['video_indices']
    dump = {'noise.video': noise_v, 'noise.audio': noise_a, 'context': ctx, 'sigmas': sched.sigmas.float()}
    xv, xa = noise_v.clone(), noise_a.clone()      # the pipeline starts from the noise itself (MiniMaxH3PrepareLatentsStep)
    mask = torch.ones(1, ctx.shape[1], dtype=torch.bool)
    for i in range(a.steps):
        sv, sa = sched.sigmas[i], asched.sigmas[i]
        last = i == a.steps - 1
        s = time.time()
        with torch.inference_mode():
            out = model(x={'image': xv, 'audio': xa}, t={'image': (1 - sv).view(1), 'audio': (1 - sa).view(1)},
                        encoder_hidden_states=ctx, encoder_hidden_states_mask=mask, return_pixels=dwd and last)
        if isinstance(out, tuple):
            out, aux = out
        uv, ua = out['image'].float(), out['audio'].float()
        x0v, x0a = xv - sv * uv, xa - sa * ua
        # the engine's sampler carries the audio as 4 x its latent (ComfyUI's H3 sampling): its estimate is 4 x0
        dump[f'x0.{i + 1:02d}.video'], dump[f'x0.{i + 1:02d}.audio'] = x0v, 4 * x0a
        step[0] += 1
        xv = sched.step(uv, sched.timesteps[i], xv, return_dict=False)[0]
        xa = asched.step(ua, asched.timesteps[i], xa, return_dict=False)[0]
        print(f'step {i + 1}/{a.steps}: sigma {float(sv):.6f} / audio {float(sa):.6f}, {time.time() - s:.0f} s (weights {t_load[0]:.0f} s so far)',
              flush=True)
        if dwd and last:
            vi = layout['video_indices']
            dump['dwd.hidden'] = caught['h'][0, vi].float().contiguous()                # [video rows, hidden], token order
            dump['dwd.features'] = caught['features'][0, vi].float().contiguous()
            row = int(caught['idx'][0, vi[0]])
            dump['dwd.temb'] = caught['temb'][row].float().contiguous()                  # the video rows' curve point [8]
            px = aux['image'].float()                                                     # [1, 3, F, 16H, 16W] Sol-normalized
            mean, std = torch.tensor(COLOR_MEAN).view(1, 3, 1, 1, 1), torch.tensor(COLOR_STD).view(1, 3, 1, 1, 1)
            img = ((px * std + mean).clamp(-1, 1) + 1) / 2
            dump['images'] = img[0].permute(1, 2, 3, 0).contiguous()                     # [F, 16H, 16W, 3] in [0, 1]
            print(f'dwd: {tuple(dump["dwd.hidden"].shape)} final hidden states -> {tuple(dump["images"].shape)} frames', flush=True)
    dump['samples.video'], dump['samples.audio'] = xv.float(), xa.float()
    dump.update(tapped)
    meta = {'seed': str(a.seed), 'shift': '12', 'audio_shift': '3', 'audio_scale': '4', 'sampler': 'piflow', 'steps': str(a.steps),
            'max_raw_t': str(MAX_RAW_T[a.steps]), 'final_step_size_scale': str(FINAL_STEP_SIZE_SCALE), 'adapter': Path(a.adapter).name,
            'pure_noise': '1'}
    save_file({k: v.contiguous() for k, v in dump.items()}, a.out, metadata=meta)
    print(f'written {a.out}; {time.time() - t0:.0f} s in all', flush=True)


if __name__ == '__main__':
    main()
