"""reference/hunyuanvideo15/ref.py: HunyuanVideo-1.5 on the CPU, from the files nextsycl serves, dumping every stage the
engine is checked against (`nextsycl video job check` with the hunyuanvideo15 engine, docs/ports/hunyuanvideo15.md).

    git clone https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5 OFF && git -C OFF checkout 60783e7   (the official code)
    python ref.py te   --te qwen2.5-vl-7b-test-q4_0.gguf --tokenizer tokenizer.json --prompt TEXT [--negative TEXT] --out DIR
    python ref.py byt5 --byt5 byt5_small_glyphxl_fp32-f16.gguf --prompt TEXT [--negative TEXT] --out DIR
    python ref.py dit  --dit transformer/480p_t2v_distilled/diffusion_pytorch_model.safetensors --off OFF --out DIR
                       [--weights q8|fp32] [--frames 9] [--height 128] [--width 224] [--steps 2] [--shift 5] [--cfg 1] [--seed 7] [--full]
    python ref.py vae  --vae vae/diffusion_pytorch_model.safetensors --off OFF --out DIR [--latents FILE.npy --frames F --height H --width W]

The stages run one at a time (each frees its weights; hardac1: `podman run --memory 40g`) and talk through DIR:
  te:   tokens.json (the template's ids, the crop), te_layer0.npy, embeds.npy [T, 3584] - Qwen2.5-VL's hidden_states[-3]
        (layer 25's output, no final norm) with the template's tokens up to and including "<|im_start|>user\n" cropped,
        as the pipeline's TextEncoder.encode with the video template (PROMPT_TEMPLATE_ENCODE_VIDEO_JSON, chat-formatted,
        the generation prompt added); neg_* for --negative
  byt5: byt5.json, byt5.npy [L, 1472] - the glyph encoder's valid rows (none when the prompt quotes nothing)
  dit:  noise.npy [F*h*w, 32] (torch.randn in bfloat16 from --seed, channels last), sigmas.npy, vec0.npy, txt0.npy (the
        token refiner's output), txt_all0.npy (the text rows the blocks see: [valid byT5 | valid Qwen] + the condition
        type embeddings), dbl0_img.npy / dbl0_txt.npy (double block 0), dbl_img.npy (after the 54 blocks), v0.npy (the
        first velocity [F*h*w, 32]), with --full latents_<i>.npy after every step (the pipeline's Euler)
  vae:  video.npy [F, H, W, 3] in [-1, 1] - AutoencoderKLConv3D.decode(latents / 1.03682) under the pipeline's
        memory_efficient_context (spatial tiles of 256 pixels, overlap 0.125) - and frame_000.png ...

The text encoder and the glyph encoder read the same GGUF files the engine does (dequantized exactly to float32: its
HunyuanImage 2.1 files, transformers' Qwen2_5_VLTextModel and T5EncoderModel). The DiT runs block by block through the
official modules (MMDoubleStreamBlock, SingleTokenRefiner, TimestepEmbedder, ByT5Mapper, VisionProjection, FinalLayer,
PatchEmbed, get_nd_rotary_pos_embed, reorder_txt_token), each block's weights read when it runs, in float32: with
--weights q8 (the default) every matrix the engine keeps as Q8_0 is rounded the way the engine rounds it (ggml's Q8_0:
blocks of 32 over an fp16 scale), so a difference from the engine is the engine's arithmetic; --weights fp32 runs the
file as it is (the quantization's own effect). Attention is SDPA over the valid tokens - upstream's default flash
attention's unpadded form (flash_attn_no_pad): the padded text rows take no part in it.
"""
import argparse
import json
import os
import re
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F

torch.set_grad_enabled(False)
torch.set_num_threads(int(os.environ.get("REF_THREADS", "0")) or (os.cpu_count() or 8))

SYSTEM = ("You are a helpful assistant. Describe the video by detailing the following aspects:         1. The main content and "
          "theme of the video.         2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects.  "
          "       3. Actions, events, behaviors temporal relationships, physical movement changes of the objects.         4. background "
          "environment, light, style and atmosphere.         5. camera angles, movements, and transitions used in the video.")
HEAD = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n"
TAIL = "<|im_end|>\n<|im_start|>assistant\n"
TEXT_LEN = 1000
BYT5_LEN = 256
SCALING = 1.03682


class Weights:
    """A GGUF file's tensors, dequantized to float32 on request"""

    def __init__(self, path):
        import gguf
        self.gguf = gguf
        self.r = gguf.GGUFReader(path)
        self.t = {t.name: t for t in self.r.tensors}

    def names(self, prefix=""):
        return [n for n in self.t if n.startswith(prefix)]

    def get(self, name):
        t = self.t[name]
        a = self.gguf.quants.dequantize(t.data, t.tensor_type)
        shape = [int(x) for x in reversed(t.shape)]
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32).reshape(shape))


def save(out, name, t):
    a = t.detach().float().contiguous().numpy() if isinstance(t, torch.Tensor) else np.asarray(t, dtype=np.float32)
    np.save(os.path.join(out, name), a.astype(np.float32))
    print(f"  {name}: {list(a.shape)}  rms {float(np.sqrt(np.mean(a.astype(np.float64) ** 2))):.4g}", flush=True)


# ------------------------------------------------------------------------------------------------ text encoder
def run_te(a):
    from transformers import PreTrainedTokenizerFast, Qwen2_5_VLTextConfig, Qwen2_5_VLTextModel
    tok = PreTrainedTokenizerFast(tokenizer_file=a.tokenizer)
    w = Weights(a.te)
    # hidden_states[-3] of 28 layers is layer 25's output: 26 layers are enough, the final norm left out
    layers = 26
    cfg = Qwen2_5_VLTextConfig(hidden_size=3584, intermediate_size=18944, num_hidden_layers=layers, num_attention_heads=28,
                               num_key_value_heads=4, rms_norm_eps=1e-6, vocab_size=152064, max_position_embeddings=128000,
                               rope_parameters={"rope_type": "default", "rope_theta": 1e6, "mrope_section": [16, 24, 24]})
    cfg._attn_implementation = "eager"
    # built without memory, then filled tensor by tensor: 26 layers in float32 are ~27 GB
    with torch.device("meta"):
        m = Qwen2_5_VLTextModel(cfg)
    m = m.to_empty(device="cpu").float().eval()
    m.rotary_emb = type(m.rotary_emb)(config=cfg)
    m.norm = torch.nn.Identity()
    per = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj",
           "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj", "attn_norm": "input_layernorm",
           "ffn_norm": "post_attention_layernorm"}
    params = dict(m.named_parameters())
    filled = set()

    def put(name, t):
        params[name].data.copy_(t)
        filled.add(name)

    put("embed_tokens.weight", w.get("token_embd.weight"))
    for n in w.names("blk."):
        _, l, role, kind = n.split(".")
        if int(l) < layers:
            put(f"layers.{l}.{per[role]}.{kind}", w.get(n))
    assert filled == set(params), set(params) - filled
    crop = len(tok(HEAD)["input_ids"])
    print(f"te: {len(filled)} tensors, crop {crop}", flush=True)

    def encode(prompt, prefix):
        # TextEncoder.apply_text_to_template: an empty prompt becomes one space
        text = HEAD + (prompt if prompt else " ") + TAIL
        ids = tok(text, truncation=True, max_length=TEXT_LEN + crop)["input_ids"]
        assert ids[:crop] == tok(HEAD)["input_ids"]
        x = torch.tensor([ids])
        pos = torch.arange(len(ids)).view(1, 1, -1).expand(3, 1, -1)
        o = m(input_ids=x, position_ids=pos, output_hidden_states=True)
        hs = o.hidden_states
        assert len(hs) == layers + 1, len(hs)
        save(a.out, f"{prefix}te_layer0.npy", hs[1][0])
        save(a.out, f"{prefix}embeds.npy", hs[-1][0, crop:])
        return ids

    meta = {"prompt": a.prompt, "ids": encode(a.prompt, ""), "crop": crop, "head": HEAD, "tail": TAIL}
    if a.negative is not None:
        meta["negative"] = a.negative
        meta["neg_ids"] = encode(a.negative, "neg_")
    json.dump(meta, open(os.path.join(a.out, "tokens.json"), "w"))


# ------------------------------------------------------------------------------------------------ glyph encoder
def quoted(prompt):
    """The pipeline's _extract_glyph_texts + MultilingualPromptFormat (no style): "..." and “...” in order, deduplicated
    when there are several"""
    texts = [m[0] or m[1] for m in re.findall(r'\"(.*?)\"|“(.*?)”', prompt)]
    texts = list(dict.fromkeys(texts)) if len(texts) > 1 else texts
    return "".join(f'Text "{t}". ' for t in texts) if texts else None


def run_byt5(a):
    from transformers import ByT5Tokenizer, T5Config, T5EncoderModel
    w = Weights(a.byt5)
    cfg = T5Config(vocab_size=1510, d_model=1472, d_kv=64, d_ff=3584, num_layers=12, num_heads=6, relative_attention_num_buckets=32,
                   relative_attention_max_distance=128, feed_forward_proj="gated-gelu", layer_norm_epsilon=1e-6, dropout_rate=0.0,
                   is_encoder_decoder=False, use_cache=False)
    m = T5EncoderModel(cfg).float().eval()
    sd = {n: w.get(n) for n in w.names()}
    sd["encoder.embed_tokens.weight"] = sd["shared.weight"]
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    tok = ByT5Tokenizer()
    meta = {}
    for prefix, prompt in [("", a.prompt)] + ([("neg_", a.negative)] if a.negative is not None else []):
        text = quoted(prompt)
        meta[f"{prefix}text"] = text
        if text is None:
            continue
        enc = tok(text, padding="max_length", max_length=BYT5_LEN, truncation=True, add_special_tokens=True, return_tensors="pt")
        ids, mask = enc.input_ids, enc.attention_mask
        meta[f"{prefix}ids"] = ids[0][mask[0].bool()].tolist()
        o = m(input_ids=ids, attention_mask=mask.float())[0]
        save(a.out, f"{prefix}byt5.npy", o[0][mask[0].bool()])
    json.dump(meta, open(os.path.join(a.out, "byt5.json"), "w"))


# ------------------------------------------------------------------------------------------------ the DiT
def official(off):
    """The official modules importable on the CPU: loguru stood in for, attention as SDPA over the valid tokens"""
    sys.path.insert(0, off)
    if "loguru" not in sys.modules:
        lg = types.ModuleType("loguru")

        class _L:
            def __getattr__(self, _):
                return lambda *x, **k: None

        lg.logger = _L()
        sys.modules["loguru"] = lg
    from hyvideo.models.transformers import hunyuanvideo_1_5_transformer as T
    from hyvideo.models.transformers.modules import token_refiner as R

    def sdpa(q, k, v, attn_mask=None, **_):
        # [B, L, H, D] each, every token valid
        if attn_mask is not None:
            assert bool(attn_mask.bool().all()), "the reference runs the valid tokens only"
        o = F.scaled_dot_product_attention(q.transpose(1, 2).float(), k.transpose(1, 2).float(), v.transpose(1, 2).float())
        return o.transpose(1, 2).reshape(q.shape[0], q.shape[1], -1)

    def joint(q, k, v, img_q_len, img_kv_len, attn_mode=None, text_mask=None, attn_param=None, block_idx=None):
        if text_mask is not None:
            assert bool(text_mask.bool().all()), "the reference runs the valid tokens only"
        return sdpa(torch.cat(q, 1), torch.cat(k, 1), torch.cat(v, 1))

    T.parallel_attention = joint
    R.attention = sdpa
    return T


def q8(w):
    """ggml's Q8_0 (quantize_row_q8_0_ref) along the last axis, back to float32: what the engine computes with"""
    shp = w.shape
    x = w.reshape(-1, 32).float()
    amax = x.abs().max(dim=1, keepdim=True).values
    d = amax / 127.0                                           # float32, as ggml
    inv = torch.where(d > 0, 1.0 / d, torch.zeros_like(d))
    v = x * inv
    q = torch.sign(v) * torch.floor(v.abs() + 0.5)            # roundf: half away from zero
    dh = d.float().half().float()
    return (q * dh).float().reshape(shp)


def quantized(name, t):
    """Whether the engine keeps this tensor as Q8_0: the double blocks' matrices"""
    return name.startswith("double_blocks.") and name.endswith(".weight") and t.dim() == 2 and t.shape[1] % 32 == 0


class Dit:
    def __init__(self, path, off, weights):
        from safetensors import safe_open
        self.f = safe_open(path, "pt")
        self.keys = list(self.f.keys())
        self.mode = weights
        self.T = off
        cfg = json.load(open(os.path.join(os.path.dirname(path), "config.json")))
        self.cfg = cfg
        self.blocks = cfg["mm_double_blocks_depth"]
        self.meanflow = cfg.get("use_meanflow", False)
        assert cfg["mm_single_blocks_depth"] == 0 and cfg["hidden_size"] == 2048
        from hyvideo.models.transformers.modules.embed_layers import PatchEmbed, TimestepEmbedder, VisionProjection
        from hyvideo.models.transformers.modules.mlp_layers import FinalLayer
        from hyvideo.models.transformers.modules.token_refiner import SingleTokenRefiner
        from hyvideo.models.text_encoders.byT5 import ByT5Mapper
        silu = torch.nn.SiLU
        H = 2048
        self.img_in = self.load(PatchEmbed([1, 1, 1], 32, H, concat_condition=True), "img_in.")
        self.time_in = self.load(TimestepEmbedder(H, silu), "time_in.")
        self.time_r_in = self.load(TimestepEmbedder(H, silu), "time_r_in.") if self.meanflow else None
        self.txt_in = self.load(SingleTokenRefiner(3584, H, 16, depth=2), "txt_in.")
        self.byt5_in = self.load(ByT5Mapper(in_dim=1472, out_dim=2048, hidden_dim=2048, out_dim1=H, use_residual=False), "byt5_in.")
        self.vision_in = self.load(VisionProjection(1152, H), "vision_in.")
        self.cond = self.tensor("cond_type_embedding.weight")
        self.final = self.load(FinalLayer(H, [1, 1, 1], 32, silu), "final_layer.")
        print(f"dit: {self.blocks} double blocks, meanflow {self.meanflow}, weights {weights}", flush=True)

    def tensor(self, name):
        t = self.f.get_tensor(name).float()
        return q8(t) if self.mode == "q8" and quantized(name, t) else t

    def part(self, prefix):
        return {n[len(prefix):]: self.tensor(n) for n in self.keys if n.startswith(prefix)}

    def load(self, module, prefix):
        module.load_state_dict(self.part(prefix), strict=True)
        return module.float().eval()

    def double(self, i):
        b = self.T.MMDoubleStreamBlock(2048, 16, mlp_width_ratio=4, mlp_act_type="gelu_tanh", attn_mode="flash", qk_norm=True,
                                       qk_norm_type="rms", qkv_bias=True)
        return self.load(b, f"double_blocks.{i}.")

    def forward(self, lat, t, txt, byt5, tr=None, vision=None, dump=None):
        """HunyuanVideo_1_5_DiffusionTransformer.forward, block by block, the valid tokens only: lat [1, 65, F, h, w]; t, tr
        [1] (x 1000); txt [1, T, 3584]; byt5 [1, L, 1472] or None; vision [1, 729, 1152] or None (T2V)"""
        _, _, tt, th, tw = lat.shape
        from hyvideo.models.transformers.modules.posemb_layers import get_nd_rotary_pos_embed
        cos, sin = get_nd_rotary_pos_embed((16, 56, 56), (tt, th, tw), theta=256, use_real=True, theta_rescale_factor=1)
        img = self.img_in(lat)
        vec = self.time_in(t)
        if tr is not None:
            vec = vec + self.time_r_in(tr)
        txt = self.txt_in(txt, t, None)
        if dump:
            dump("vec0.npy", vec[0])
            dump("txt0.npy", txt[0])
        txt = txt + self.cond[0]
        parts = [txt]
        if byt5 is not None:
            parts.insert(0, self.byt5_in(byt5) + self.cond[1])
        if vision is not None:
            parts.insert(0, self.vision_in(vision) + self.cond[2])
        txt = torch.cat(parts, 1)
        if dump:
            dump("txt_all0.npy", txt[0])
        for i in range(self.blocks):
            img, txt = self.double(i)(img, txt, vec, (cos, sin), None, None)
            if dump and i == 0:
                dump("dbl0_img.npy", img[0])
                dump("dbl0_txt.npy", txt[0])
        if dump:
            dump("dbl_img.npy", img[0])
        out = self.final(img, vec)                                # [1, F h w, 32]
        return out


def tokens(x):
    """[1, C, F, h, w] -> [F h w, C]"""
    return x[0].permute(1, 2, 3, 0).reshape(-1, x.shape[1])


def volume(x, f, h, w):
    return x.reshape(f, h, w, -1).permute(3, 0, 1, 2)[None]


def text_inputs(out, neg):
    p = "neg_" if neg else ""
    e = torch.from_numpy(np.load(os.path.join(out, f"{p}embeds.npy")))[None]
    b5 = os.path.join(out, f"{p}byt5.npy")
    b = torch.from_numpy(np.load(b5))[None] if os.path.exists(b5) else None
    return e, b


def run_dit(a):
    T = official(a.off)
    d = Dit(a.dit, T, a.weights)
    f = (a.frames - 1) // 4 + 1
    h, w = a.height // 16, a.width // 16
    gen = torch.Generator(device="cpu").manual_seed(a.seed)
    noise = torch.randn((1, 32, f, h, w), generator=gen, dtype=torch.bfloat16).float()
    save(a.out, "noise.npy", tokens(noise))
    sig = torch.linspace(1, 0, a.steps + 1)
    sig = (a.shift * sig) / (1 + (a.shift - 1) * sig)
    save(a.out, "sigmas.npy", sig)
    ts = (sig[:-1] * 1000).float()
    pos = text_inputs(a.out, False)
    cfg = a.cfg if a.cfg > 1 else None
    neg = text_inputs(a.out, True) if cfg else None
    json.dump({"frames": a.frames, "height": a.height, "width": a.width, "latent": [f, h, w], "steps": a.steps, "shift": a.shift,
               "cfg": cfg, "seed": a.seed, "meanflow": d.meanflow, "weights": a.weights}, open(os.path.join(a.out, "dit.json"), "w"))
    lat = noise
    # T2V: the condition latents zero, the mask zero (the pipeline's _prepare_cond_latents)
    cond = torch.zeros(1, 33, f, h, w)
    for i in range(a.steps if a.full else 1):
        t = ts[i:i + 1]
        tr = (ts[i + 1:i + 2] if i + 1 < a.steps else torch.zeros(1)) if d.meanflow else None
        dump = (lambda n, v: save(a.out, n, v)) if i == 0 else None
        x = torch.cat([lat, cond], 1)
        v = d.forward(x, t, pos[0], pos[1], tr=tr, dump=dump)
        if cfg:
            vn = d.forward(x, t, neg[0], neg[1], tr=tr)
            v = vn + cfg * (v - vn)
        if i == 0:
            save(a.out, "v0.npy", v[0])
        lat = lat + (sig[i + 1] - sig[i]) * volume(v[0], f, h, w)
        if a.full:
            save(a.out, f"latents_{i}.npy", tokens(lat))
        print(f"step {i + 1}/{a.steps}", flush=True)


# ------------------------------------------------------------------------------------------------ the VAE
def vae_model(a):
    official(a.off)
    from hyvideo.models.autoencoders.hunyuanvideo_15_vae import AutoencoderKLConv3D
    from safetensors.torch import load_file
    cfg = json.load(open(os.path.join(os.path.dirname(a.vae), "config.json")))
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    m = AutoencoderKLConv3D(**cfg)
    m.load_state_dict(load_file(a.vae), strict=True)
    m = m.float().eval()
    # the pipeline: set_tile_sample_min_size(256, 0.125) on a card above 21 GB, decode under memory_efficient_context
    m.set_tile_sample_min_size(256, 0.125)
    return m


def run_vae(a):
    m = vae_model(a)
    if a.latents:
        lat = torch.from_numpy(np.load(a.latents))
        f, h, w = (a.frames - 1) // 4 + 1, a.height // 16, a.width // 16
    else:
        meta = json.load(open(os.path.join(a.out, "dit.json")))
        f, h, w = meta["latent"]
        lat = torch.from_numpy(np.load(os.path.join(a.out, f"latents_{meta['steps'] - 1}.npy")))
    z = volume(lat, f, h, w) / SCALING
    with m.memory_efficient_context():
        v = m.decode(z, return_dict=False)[0]                 # [1, 3, F, H, W]
    v = v[0].permute(1, 2, 3, 0)
    save(a.out, "video.npy", v)
    from PIL import Image
    for i in range(v.shape[0]):
        Image.fromarray(((v[i] / 2 + 0.5).clamp(0, 1) * 255).round().byte().numpy()).save(os.path.join(a.out, f"frame_{i:03d}.png"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["te", "byt5", "dit", "vae"])
    p.add_argument("--te")
    p.add_argument("--tokenizer")
    p.add_argument("--byt5")
    p.add_argument("--dit")
    p.add_argument("--vae")
    p.add_argument("--off", help="the official code (a clone of Tencent-Hunyuan/HunyuanVideo-1.5 at 60783e7)")
    p.add_argument("--prompt", default="")
    p.add_argument("--negative")
    p.add_argument("--out", required=True)
    p.add_argument("--weights", default="q8", choices=["q8", "fp32"])
    p.add_argument("--frames", type=int, default=9)
    p.add_argument("--height", type=int, default=128)
    p.add_argument("--width", type=int, default=224)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--cfg", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--full", action="store_true")
    p.add_argument("--latents")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    {"te": run_te, "byt5": run_byt5, "dit": run_dit, "vae": run_vae}[a.stage](a)


if __name__ == "__main__":
    main()
