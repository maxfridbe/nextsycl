"""reference/hunyuanimage21/ref.py: HunyuanImage 2.1 on the CPU, from the SAME GGUF files nextsycl serves, dumping every
stage the engine is checked against (`nextsycl image check DIR --stages ...`).

    sh fetch.sh OFF                                   the official model code (Tencent-Hunyuan/HunyuanImage-2.1, pinned)
    python ref.py te   --te qwen2.5-vl-7b-test-q4_0.gguf --tokenizer tokenizer.json --prompt TEXT [--negative TEXT] --out DIR
    python ref.py byt5 --byt5 byt5_small_glyphxl_fp32-f16.gguf --prompt TEXT --out DIR
    python ref.py dit  --dit hunyuanimage2.1-distilled-v2-q8_0.gguf --off OFF --out DIR --size 512 --steps 8 [--shift 4]
                       [--guidance 3.25] [--cfg 3.5] [--seed 7] [--full]
    python ref.py vae  --vae pig_hunyuan_image_vae_fp32-f16.gguf --off OFF --out DIR [--latents FILE.npy] [--size 512]

The stages run one at a time (each frees its weights) and talk through DIR:
  te:   tokens.json (the template's ids, the crop), embeds.npy [T, 3584] - Qwen2.5-VL's hidden_states[-3] (layer 25's
        output, no final norm) with the template's first 34 tokens cropped, as the pipeline's TextEncoder.encode;
        neg_embeds.npy for --negative (the base model's guidance; "" is the pipeline's default)
  byt5: byt5.json (the quoted text, its ids), byt5.npy [L, 1472] - the glyph encoder's output for the valid tokens
        (nothing when the prompt quotes nothing); neg_byt5.npy likewise
  dit:  noise.npy [N, 64] (latents channels last, from --seed), sigmas.npy, vec0.npy (step 0's modulation input),
        txt0.npy (the token refiner's output, step 0), dbl0_img.npy / dbl0_txt.npy (double block 0's outputs),
        dbl_img.npy (after the 20 double blocks), v0.npy (step 0's velocity [N, 64]), with --full latents_<i>.npy
        after every step (Euler, the static shift; MeanFlow's next timestep when the file has time_r_in)
  vae:  image.npy [H, W, 3] in [-1, 1] (the decoder on latents / 0.75289), image.png

Weights are the GGUF tensors dequantized exactly (Q8_0 / Q4_0 / Q6_K / F16 / BF16 -> float32) and the activations
float32, so a difference from the engine is the engine's rounding. The DiT runs block by block through the official
modules (MMDoubleStreamBlock, MMSingleStreamBlock, SingleTokenRefiner, TimestepEmbedder, ByT5Mapper, FinalLayer,
get_nd_rotary_pos_embed, reorder_txt_token), each block's weights loaded when it runs: 17B parameters never sit in
memory at once. Flash attention's unpadded form is SDPA with the key padding mask (padded rows zero, as pad_input).
The text encoder is transformers' Qwen2_5_VLTextModel (text only: the three M-RoPE axes equal), the glyph encoder
transformers' T5EncoderModel with byT5-small's configuration.
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
torch.set_num_threads(os.cpu_count() or 8)

TEMPLATE = ("<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial "
            "relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>")
CROP = 34
MAX_TOKENS = 1000 + CROP
SCALING = 0.75289


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

    def sub(self, prefix):
        return {n: self.get(n) for n in self.names(prefix)}


def save(out, name, t):
    a = t.detach().float().contiguous().numpy() if isinstance(t, torch.Tensor) else np.asarray(t, dtype=np.float32)
    np.save(os.path.join(out, name), a.astype(np.float32))
    print(f"  {name}: {list(a.shape)}  rms {float(np.sqrt(np.mean(a.astype(np.float64) ** 2))):.4g}", flush=True)


# ------------------------------------------------------------------------------------------------ text encoder
def run_te(a):
    from transformers import PreTrainedTokenizerFast, Qwen2_5_VLTextConfig, Qwen2_5_VLTextModel
    tok = PreTrainedTokenizerFast(tokenizer_file=a.tokenizer)
    w = Weights(a.te)
    cfg = Qwen2_5_VLTextConfig(hidden_size=3584, intermediate_size=18944, num_hidden_layers=28, num_attention_heads=28,
                               num_key_value_heads=4, rms_norm_eps=1e-6, vocab_size=152064, max_position_embeddings=128000,
                               rope_parameters={"rope_type": "default", "rope_theta": 1e6, "mrope_section": [16, 24, 24]})
    cfg._attn_implementation = "eager"
    m = Qwen2_5_VLTextModel(cfg).float().eval()
    names = {"token_embd.weight": "embed_tokens.weight", "output_norm.weight": "norm.weight"}
    per = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj",
           "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj", "attn_norm": "input_layernorm",
           "ffn_norm": "post_attention_layernorm"}
    sd = {}
    for n in w.names():
        if n.startswith("blk."):
            _, l, role, kind = n.split(".")
            sd[f"layers.{l}.{per[role]}.{kind}"] = w.get(n)
        elif n in names:
            sd[names[n]] = w.get(n)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not missing, missing
    print(f"te: {len(sd)} tensors ({len(unexpected)} unused)", flush=True)

    def encode(prompt, prefix):
        ids = tok(TEMPLATE.format(prompt), truncation=True, max_length=MAX_TOKENS)["input_ids"]
        x = torch.tensor([ids])
        pos = torch.arange(len(ids)).view(1, 1, -1).expand(3, 1, -1)
        o = m(input_ids=x, position_ids=pos, output_hidden_states=True)
        hs = o.hidden_states
        assert len(hs) == 29, len(hs)
        save(a.out, f"{prefix}te_layer0.npy", hs[1][0])
        save(a.out, f"{prefix}embeds.npy", hs[-3][0, CROP:])
        return ids

    ids = encode(a.prompt, "")
    meta = {"prompt": a.prompt, "ids": ids, "crop": CROP}
    if a.negative is not None:
        meta["negative"] = a.negative
        meta["neg_ids"] = encode(a.negative, "neg_")
    json.dump(meta, open(os.path.join(a.out, "tokens.json"), "w"))


# ------------------------------------------------------------------------------------------------ glyph encoder
def quoted(prompt):
    """The pipeline's _encode_glyph: the quoted pieces, formatted as MultilingualPromptFormat with no style"""
    texts = re.findall(r'\"(.*?)\"', prompt) + re.findall(r'‘(.*?)’', prompt) + re.findall(r'“(.*?)”', prompt)
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
        enc = tok(text, padding="max_length", max_length=128, truncation=True, add_special_tokens=True, return_tensors="pt")
        ids, mask = enc.input_ids, enc.attention_mask
        meta[f"{prefix}ids"] = ids[0][mask[0].bool()].tolist()
        o = m(input_ids=ids, attention_mask=mask.float())[0]
        save(a.out, f"{prefix}byt5.npy", o[0][mask[0].bool()])
    json.dump(meta, open(os.path.join(a.out, "byt5.json"), "w"))


# ------------------------------------------------------------------------------------------------ the DiT
def official(off):
    """The official modules, with flash attention's unpadded entry points as SDPA"""
    sys.path.insert(0, off)
    shim = types.ModuleType("hyimage.models.hunyuan.modules.flash_attn_no_pad")

    def flash_attn_no_pad(qkv, key_padding_mask, causal=False, dropout_p=0.0, softmax_scale=None, deterministic=False):
        q, k, v = qkv.unbind(2)                                  # [B, S, H, D]
        q, k, v = (t.transpose(1, 2).float() for t in (q, k, v))
        mask = None if key_padding_mask is None else key_padding_mask.bool()[:, None, None, :]
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal, scale=softmax_scale)
        o = o.transpose(1, 2)
        if key_padding_mask is not None:                         # pad_input: the padded rows are zeros
            o = o * key_padding_mask.bool()[:, :, None, None]
        return o.to(qkv.dtype)

    def get_cu_seqlens(text_mask, img_len):
        return None, None

    shim.flash_attn_no_pad = flash_attn_no_pad
    shim.get_cu_seqlens = get_cu_seqlens
    sys.modules[shim.__name__] = shim
    from hyimage.models.hunyuan.modules import hunyuanimage_dit as dit
    return dit


def load(module, sd):
    module.load_state_dict(sd, strict=True)
    return module.float().eval()


class Dit:
    def __init__(self, path, off):
        self.w = Weights(path)
        self.o = official(off)
        from hyimage.models.hunyuan.modules.embed_layers import PatchEmbed2D, TimestepEmbedder
        from hyimage.models.hunyuan.modules.mlp_layers import FinalLayer
        from hyimage.models.hunyuan.modules.models import MMDoubleStreamBlock, MMSingleStreamBlock
        from hyimage.models.hunyuan.modules.posemb_layers import get_nd_rotary_pos_embed
        from hyimage.models.hunyuan.modules.token_refiner import SingleTokenRefiner
        from hyimage.models.text_encoder.byT5 import ByT5Mapper
        self.c = dict(PatchEmbed2D=PatchEmbed2D, TimestepEmbedder=TimestepEmbedder, FinalLayer=FinalLayer, Double=MMDoubleStreamBlock,
                      Single=MMSingleStreamBlock, rope=get_nd_rotary_pos_embed, Refiner=SingleTokenRefiner, ByT5Mapper=ByT5Mapper)
        self.meanflow = "time_r_in.mlp.0.weight" in self.w.t
        self.guidance = "guidance_in.mlp.0.weight" in self.w.t
        self.doubles = len({n.split(".")[1] for n in self.w.names("double_blocks.")})
        self.singles = len({n.split(".")[1] for n in self.w.names("single_blocks.")})
        print(f"dit: {self.doubles} double, {self.singles} single blocks; meanflow {self.meanflow}, guidance {self.guidance}", flush=True)
        # the small parts stay loaded (every step needs them)
        silu = torch.nn.SiLU
        self.img_in = load(self.c["PatchEmbed2D"]([1, 1], 64, 3584), self.part("img_in."))
        self.time_in = load(self.c["TimestepEmbedder"](3584, silu), self.part("time_in."))
        self.time_r_in = load(self.c["TimestepEmbedder"](3584, silu), self.part("time_r_in.")) if self.meanflow else None
        self.guidance_in = load(self.c["TimestepEmbedder"](3584, silu), self.part("guidance_in.")) if self.guidance else None
        self.txt_in = load(self.c["Refiner"](3584, 3584, 28, depth=2), self.part("txt_in."))
        self.byt5_in = load(self.c["ByT5Mapper"](1472, 2048, 2048, 3584, use_residual=False), self.part("byt5_in."))
        self.final = load(self.c["FinalLayer"](3584, [1, 1], 64, silu), self.part("final_layer."))

    def part(self, prefix):
        return {n[len(prefix):]: v for n, v in self.w.sub(prefix).items()}

    def double(self, i):
        sd = self.o.convert_hunyuan_dict_for_tensor_parallel(self.w.sub(f"double_blocks.{i}."))
        sd = {n[len(f"double_blocks.{i}."):]: v for n, v in sd.items()}
        b = self.c["Double"](3584, 28, mlp_width_ratio=4, mlp_act_type="gelu_tanh", qk_norm=True, qk_norm_type="rms", qkv_bias=True)
        return load(b, sd)

    def single(self, i):
        sd = self.o.convert_hunyuan_dict_for_tensor_parallel(self.w.sub(f"single_blocks.{i}."))
        sd = {n[len(f"single_blocks.{i}."):]: v for n, v in sd.items()}
        b = self.c["Single"](3584, 28, mlp_width_ratio=4, mlp_act_type="gelu_tanh", qk_norm=True, qk_norm_type="rms")
        return load(b, sd)

    def forward(self, lat, t, txt, txt_mask, byt5, byt5_mask, tr=None, guidance=None, dump=None):
        """HYImageDiffusionTransformer.forward, block by block: lat [1, 64, h, w]; t, tr, guidance [1] (x 1000)"""
        _, _, th, tw = lat.shape
        cos, sin = self.c["rope"]([64, 64], (th, tw), theta=256, use_real=True, theta_rescale_factor=1)
        img = self.img_in(lat)
        vec = self.time_in(t)
        if tr is not None:
            vec = (vec + self.time_r_in(tr)) / 2
        if self.guidance:
            vec = vec + self.guidance_in(guidance)
        txt = self.txt_in(txt, t, txt_mask)
        if dump:
            dump("vec0.npy", vec[0])
            dump("txt0.npy", txt[0][txt_mask[0].bool()])
        byt5_txt = self.byt5_in(byt5)
        txt, mask = self.o.HYImageDiffusionTransformer.reorder_txt_token(None, byt5_txt, txt, byt5_mask, txt_mask)
        for i in range(self.doubles):
            img, txt = self.double(i)(img, txt, vec, (cos, sin), mask, None, None)
            if dump and i == 0:
                dump("dbl0_img.npy", img[0])
                dump("dbl0_txt.npy", txt[0][mask[0].bool()])
        if dump:
            dump("dbl_img.npy", img[0])
        x = torch.cat((img, txt), 1)
        n = img.shape[1]
        for i in range(self.singles):
            x = self.single(i)(x, vec, txt.shape[1], (cos, sin), mask, None, None)
        out = self.final(x[:, :n], vec)
        return self.o.HYImageDiffusionTransformer.unpatchify_2d(types.SimpleNamespace(unpatchify_channels=64, patch_size=[1, 1]), out, th, tw)


def to_tokens(x):
    """[1, 64, h, w] -> [h * w, 64]"""
    return x[0].permute(1, 2, 0).reshape(-1, x.shape[1])


def from_tokens(x, h, w):
    return x.reshape(h, w, -1).permute(2, 0, 1)[None]


def text_inputs(out, neg):
    p = "neg_" if neg else ""
    e = torch.from_numpy(np.load(os.path.join(out, f"{p}embeds.npy")))[None]
    b5 = os.path.join(out, f"{p}byt5.npy")
    if os.path.exists(b5):
        b = torch.from_numpy(np.load(b5))
        byt5 = torch.zeros(1, 128, 1472)
        byt5[0, :b.shape[0]] = b
        bmask = torch.zeros(1, 128, dtype=torch.int64)
        bmask[0, :b.shape[0]] = 1
    else:
        byt5, bmask = torch.zeros(1, 128, 1472), torch.zeros(1, 128, dtype=torch.int64)
    return e, torch.ones(1, e.shape[1], dtype=torch.int64), byt5, bmask


def run_dit(a):
    d = Dit(a.dit, a.off)
    h = w = a.size // 32
    gen = np.random.default_rng(a.seed)
    noise = gen.standard_normal((h * w, 64)).astype(np.float32)
    save(a.out, "noise.npy", noise)
    sig = torch.linspace(1, 0, a.steps + 1)
    sig = (a.shift * sig) / (1 + (a.shift - 1) * sig)
    save(a.out, "sigmas.npy", sig)
    ts = sig[:-1] * 1000
    pos = text_inputs(a.out, False)
    cfg = a.cfg if (not d.guidance and a.cfg > 1) else None
    neg = text_inputs(a.out, True) if cfg else None
    json.dump({"size": a.size, "steps": a.steps, "shift": a.shift, "guidance": a.guidance, "cfg": cfg, "seed": a.seed,
               "meanflow": d.meanflow}, open(os.path.join(a.out, "dit.json"), "w"))
    lat = from_tokens(torch.from_numpy(noise), h, w)
    g = torch.tensor([a.guidance * 1000.0]) if d.guidance else None
    for i in range(a.steps if a.full else 1):
        t = ts[i:i + 1]
        tr = (ts[i + 1:i + 2] if i + 1 < a.steps else torch.zeros(1)) if d.meanflow else None
        dump = (lambda n, v: save(a.out, n, v)) if i == 0 else None
        v = d.forward(lat, t, *pos, tr=tr, guidance=g, dump=dump)
        if cfg:
            vn = d.forward(lat, t, *neg, tr=tr, guidance=g)
            v = vn + cfg * (v - vn)
        if i == 0:
            save(a.out, "v0.npy", to_tokens(v))
        lat = lat - (sig[i] - sig[i + 1]) * v
        if a.full:
            save(a.out, f"latents_{i}.npy", to_tokens(lat))
        print(f"step {i + 1}/{a.steps}", flush=True)


# ------------------------------------------------------------------------------------------------ the VAE
def run_vae(a):
    sys.path.insert(0, a.off)
    from hyimage.models.vae.hunyuanimage_vae import Decoder
    w = Weights(a.vae)
    dec = Decoder(z_channels=64, out_channels=3, block_out_channels=list(reversed([128, 256, 512, 512, 1024, 1024])), num_res_blocks=2,
                  ffactor_spatial=32)
    sd = {n[len("decoder."):]: v for n, v in w.sub("decoder.").items()}
    dec.load_state_dict(sd, strict=True)
    dec = dec.float().eval()
    if a.latents:
        lat = torch.from_numpy(np.load(a.latents))
    else:
        steps = json.load(open(os.path.join(a.out, "dit.json")))["steps"]
        lat = torch.from_numpy(np.load(os.path.join(a.out, f"latents_{steps - 1}.npy")))
    h = w_ = a.size // 32
    z = from_tokens(lat, h, w_) / SCALING
    img = dec(z)[0].permute(1, 2, 0)
    save(a.out, "image.npy", img)
    from PIL import Image
    Image.fromarray(((img / 2 + 0.5).clamp(0, 1) * 255).round().byte().numpy()).save(os.path.join(a.out, "image.png"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["te", "byt5", "dit", "vae"])
    p.add_argument("--te")
    p.add_argument("--tokenizer")
    p.add_argument("--byt5")
    p.add_argument("--dit")
    p.add_argument("--vae")
    p.add_argument("--off", help="the official code (fetch.sh)")
    p.add_argument("--prompt", default="")
    p.add_argument("--negative")
    p.add_argument("--out", required=True)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--shift", type=float, default=4.0)
    p.add_argument("--guidance", type=float, default=3.25)
    p.add_argument("--cfg", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--full", action="store_true")
    p.add_argument("--latents")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    {"te": run_te, "byt5": run_byt5, "dit": run_dit, "vae": run_vae}[a.stage](a)


if __name__ == "__main__":
    main()
