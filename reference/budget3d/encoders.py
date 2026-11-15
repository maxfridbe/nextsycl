"""reference/budget3d/encoders.py: Idol's picture side on the CPU - its framing (idol/preprocess.py, BiRefNet) and the
two image encoders' code paths - dumped for `nextsycl model3d check DIR --stages pre,enc`.

The real encoders are gated on Hugging Face (DINOv3 ViT-H+/16, the FLUX.2 VAE). Without them this checks the code,
not the weights: it builds transformers' DINOv3ViTModel with ViT-H+/16's configuration (Budget3D's
scripts/convert_dinov3_to_hf.py: 1280 wide, 32 layers, 20 heads, SwiGLU 5120, 4 registers) and diffusers'
AutoencoderKLFlux2 with its default (FLUX.2) configuration, both seeded random, saves their weights as safetensors (the
files the engine is pointed at: --opt-dino / --opt-vae) and dumps what they make of the framed picture, in float32
(Idol runs DINOv3 and the VAE in bfloat16), the VAE's posterior mean (Budget3D's deterministic encoder). With
--dino / --vae the real files are used instead.

    python -I encoders.py OUT --code CODE --picture P [--birefnet model.safetensors] [--dino F] [--vae F] [--mode synthetic]

Dumps: picture.png (the input, as PNG), pre.npy [1024, 1024, 3] (the framed picture, 0..255), enc_dino.npy [4101, 1280], enc_vae.npy [4101, 128];
dino.safetensors / vae.safetensors when random.
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--code", required=True)
    ap.add_argument("--picture", required=True)
    ap.add_argument("--birefnet", default=None, help="BiRefNet's weights (ZhengPeng7/BiRefNet model.safetensors or Idol's birefnet.safetensors)")
    ap.add_argument("--dino", default=None)
    ap.add_argument("--vae", default=None)
    ap.add_argument("--mode", default="synthetic")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, "stubs"))
    sys.path.insert(0, os.path.join(a.code, "idol-sdk"))
    os.makedirs(a.out, exist_ok=True)

    import numpy as np
    import torch
    from PIL import Image
    from safetensors.torch import load_file, save_file
    torch.set_grad_enabled(False)
    torch.manual_seed(0)

    from idol.models.birefnet import BiRefNet
    from idol.preprocess import CROP_MARGIN, preprocess_image
    remover = None
    if a.birefnet:
        remover = BiRefNet()
        missing, unexpected = remover.load_state_dict(load_file(a.birefnet), strict=False)
        assert not missing, f"BiRefNet weights missing: {missing[:4]}"
        print(f"[enc] BiRefNet: {len(unexpected)} training-only tensors of the file unused", flush=True)
        remover = remover.eval().float()
    pic = Image.open(a.picture)
    pic.save(os.path.join(a.out, "picture.png"))
    with open(os.path.join(a.out, "framing.json"), "w") as f:
        f.write('{"margin": %r}' % CROP_MARGIN[a.mode])
    pre = preprocess_image(pic, background_remover=remover, image_size=1024, crop_margin=CROP_MARGIN[a.mode])
    arr = np.array(pre).astype(np.float32)
    np.save(os.path.join(a.out, "pre.npy"), arr)
    pixels = torch.from_numpy(arr / 255.0).permute(2, 0, 1)[None].float()

    from transformers import DINOv3ViTConfig, DINOv3ViTModel
    if a.dino:
        dino = DINOv3ViTModel.from_pretrained(os.path.dirname(a.dino), dtype=torch.float32)
    else:
        cfg = DINOv3ViTConfig(patch_size=16, hidden_size=1280, intermediate_size=5120, num_hidden_layers=32, num_attention_heads=20,
                              num_register_tokens=4, use_gated_mlp=True, hidden_act="silu", layerscale_value=0.1)
        dino = DINOv3ViTModel(cfg)
        save_file({k: v.contiguous() for k, v in dino.state_dict().items()}, os.path.join(a.out, "dino.safetensors"))
    dino = dino.eval().float()
    mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    tok = dino(pixel_values=(pixels - mean) / std).last_hidden_state.float()
    tok = torch.nn.functional.layer_norm(tok, tok.shape[-1:])
    np.save(os.path.join(a.out, "enc_dino.npy"), tok[0].numpy().astype(np.float32))
    print(f"[enc] DINOv3 tokens {tuple(tok.shape)}", flush=True)
    del dino

    from diffusers.models import AutoencoderKLFlux2
    if a.vae:
        vae = AutoencoderKLFlux2.from_pretrained(os.path.dirname(a.vae), dtype=torch.float32)
    else:
        vae = AutoencoderKLFlux2()
        # the batch norm's statistics are buffers a fresh model leaves at 0 and 1: something to check against
        vae.bn.running_mean.normal_(0, 0.5)
        vae.bn.running_var.uniform_(0.5, 2.0)
        save_file({k: v.contiguous() for k, v in vae.state_dict().items() if not k.endswith("num_batches_tracked")},
                  os.path.join(a.out, "vae.safetensors"))
    vae = vae.eval().float()
    lat = vae.encode(pixels * 2 - 1).latent_dist.mode()
    b, c, h, w = lat.shape
    lat = lat.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)
    m = vae.bn.running_mean.view(1, -1, 1, 1)
    s = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)
    t = ((lat - m) / s).flatten(2).transpose(1, 2)
    t = torch.cat([torch.zeros(b, 5, t.shape[2]), t], dim=1)
    np.save(os.path.join(a.out, "enc_vae.npy"), t[0].numpy().astype(np.float32))
    print(f"[enc] FLUX.2 VAE tokens {tuple(t.shape)}", flush=True)


if __name__ == "__main__":
    main()
