# refvis.py SIDECAR.gguf IMAGE OUT_PREFIX [f32|bf16] [ENGINE_PREFIX|-] [INFERENCE_DIR]: DeepSeek's own preprocessing
# (image_processor.load_image), ViT and aligner (vision.py, from the 1017 release's inference/ folder: INFERENCE_DIR,
# default .) on the vision sidecar's weights, in float32 or bf16. Writes OUT_PREFIX.patches.f32, .vit.f32, .rows.f32 and
# compares them with the engine's (NS_DS41_VISION_DUMP=ENGINE_PREFIX) when given. Needs torch, numpy, pillow, gguf.
import sys, types, numpy as np, torch
sys.path.insert(0, sys.argv[6] if len(sys.argv) > 6 else '.')
from gguf import GGUFReader
import vision, image_processor as ip

side, img, out = sys.argv[1], sys.argv[2], sys.argv[3]
dt = torch.bfloat16 if len(sys.argv) > 4 and sys.argv[4] == 'bf16' else torch.float32
r = GGUFReader(side)
W = {}
for t in r.tensors:
    shape = [int(x) for x in reversed(t.shape)]
    raw = np.asarray(t.data)
    if t.tensor_type.name == 'BF16':
        a = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    else:
        a = raw.view(np.float32) if raw.dtype != np.float32 else raw
    W[t.name] = torch.from_numpy(a.reshape(shape).copy())
args = types.SimpleNamespace(vision_dim=1024, vision_n_heads=16, vision_inter_dim=2816, vision_patch_size=14, vision_rope_theta=10000.0,
                             vision_n_layers=32, vision_downsample_ratio=3, dim=5120, vision_max_n_token=1024, vision_min_pixels=544 * 544,
                             vision_max_wh_ratio=None)
vit = vision.ViT(args); al = vision.Aligner(args)
sd = {'patch_embed.proj.weight': W['v.patch_embd.weight'], 'patch_embed.proj.bias': W['v.patch_embd.bias'], 'norm.weight': W['v.output_norm.weight']}
for l in range(32):
    p = f'v.blk.{l}.'
    sd.update({f'blocks.{l}.norm1.weight': W[p + 'attn_norm.weight'], f'blocks.{l}.norm2.weight': W[p + 'ffn_norm.weight'],
               f'blocks.{l}.attn.wqkv.weight': W[p + 'attn_qkv.weight'], f'blocks.{l}.attn.wqkv.bias': W[p + 'attn_qkv.bias'],
               f'blocks.{l}.attn.wo.weight': W[p + 'attn_output.weight'], f'blocks.{l}.attn.wo.bias': W[p + 'attn_output.bias'],
               f'blocks.{l}.mlp.w1.weight': W[p + 'ffn_gate_up.weight'], f'blocks.{l}.mlp.w2.weight': W[p + 'ffn_down.weight']})
vit.load_state_dict(sd)
al.load_state_dict({'w1.weight': W['mm.0.weight'], 'w1.bias': W['mm.0.bias'], 'w2.weight': W['mm.1.weight'], 'w2.bias': W['mm.1.bias']})
vit = vit.to(dt); al = al.to(dt)
patches, vh, vw, lh, lw = ip.load_image({'url': img}, args)
print('plan', vh, vw, lh, lw, 'span', ip.num_image_tokens(lh, lw))
torch.set_num_threads(16)
with torch.inference_mode():
    x = patches.to(dt)
    v = vit(x, vh, vw)
    rows = al(v, vh, vw)
res = {'patches': patches.float().reshape(vh * vw, -1).numpy(), 'vit': v.float().numpy(), 'rows': rows.float().numpy()}
for k, a in res.items():
    a.astype(np.float32).tofile(f'{out}.{k}.f32')
if len(sys.argv) > 5 and sys.argv[5] != '-':
    eng = sys.argv[5]
    for k, a in res.items():
        b = np.fromfile(f'{eng}.{k}.f32', dtype=np.float32)
        a = a.reshape(-1)
        if b.size != a.size:
            print(k, 'sizes differ', a.size, b.size); continue
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
        rel = float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-30))
        print(f'{k:8s} cos {cos:.6f} rel err {rel:.3e} max |diff| {float(np.abs(a - b).max()):.3e}')
