"""MiniMax H3's text encoder in ComfyUI (comfy/text_encoders/minimax.py) with pictures in the prompt - the reference for
`nextsycl video job encode --pictures A,B --check DUMP`: the conditioning [1, L, 5120] and the token tags of
ComfyUI's image-to-video presentation ("<Picture i>: " <vision block> ... <prompt>).

    python te_ref.py --te TE.gguf --visual VISUAL.safetensors --picture a.png [--picture b.png] --prompt "..." --out dump.safetensors
                     [--embeddings DIR]   (`embedding:NAME` in the prompt)

Pictures are taken at their own size (sides multiples of 32, so neither side resizes them). Runs where ComfyUI and
ComfyUI-GGUF import - the H3 reference image: ComfyUI at /comfy, ComfyUI-GGUF at /pkgs/comfyui_gguf (or --comfy,
--pkgs). The text encoder is the llama.cpp-quantized 32B (keys already ComfyUI's); its own visual tensors are
dropped (incomplete: no deepstack mergers) for those of --visual.
"""
import argparse
import sys
import time


def load_gguf_native(path):
    """A GGUF whose keys are ComfyUI's (arch-less), as ComfyUI-GGUF's loader wraps tensors"""
    import gguf
    import torch
    from comfyui_gguf.ops import GGMLTensor
    from comfyui_gguf.dequant import dequantize_tensor
    reader = gguf.GGUFReader(path)
    sd = {}
    for t in reader.tensors:
        tt = torch.from_numpy(t.data)
        shape = torch.Size(tuple(int(v) for v in reversed(t.shape)))
        if t.tensor_type in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
            tt = tt.view(*shape)
        sd[t.name] = GGMLTensor(tt, tensor_type=t.tensor_type, tensor_shape=shape)
        if len(shape) <= 1 and t.tensor_type == gguf.GGMLQuantizationType.BF16:
            sd[t.name] = dequantize_tensor(sd[t.name], dtype=torch.float32)
    return sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--te", required=True)
    ap.add_argument("--visual", required=True)
    ap.add_argument("--picture", action="append", default=[])
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--embeddings", default=None, help="the directory `embedding:NAME` in the prompt reads from")
    ap.add_argument("--comfy", default="/comfy")
    ap.add_argument("--pkgs", default="/pkgs")
    a = ap.parse_args()
    sys.path.insert(0, a.comfy)
    sys.path.insert(0, a.pkgs)

    import numpy as np
    import torch
    import comfy.cli_args
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()) and not torch.cuda.is_available():
        comfy.cli_args.args.cpu = True  # a CPU-only torch: ComfyUI would reach for CUDA otherwise
    from PIL import Image
    from safetensors.torch import load_file, save_file
    import comfy.sd
    import comfy.model_management as mm
    import comfy.model_patcher
    import comfy.text_encoders.minimax as mmte
    from comfyui_gguf.ops import GGMLOps
    from comfyui_gguf.dequant import dequantize_tensor

    t0 = time.time()
    sd = load_gguf_native(a.te)
    k = "model.embed_tokens.weight"
    if hasattr(sd[k], "tensor_type"):
        sd[k] = dequantize_tensor(sd[k], dtype=torch.float32)
    sd = {n: v for n, v in sd.items() if not n.startswith("visual.")}
    sd.update(load_file(a.visual))
    ops = GGMLOps()
    ops.Linear.dequant_dtype = None
    ops.Linear.patch_dtype = None
    # half on a GPU; float32 on the CPU (half matrix products there are slow and no closer)
    dt = torch.float32 if comfy.cli_args.args.cpu else torch.float16
    te = mmte.MiniMaxH3TEModel(device="cpu", dtype=dt, model_options={"custom_operations": ops})
    missing, unexpected = te.load_sd(sd)
    print(f"te loaded in {time.time() - t0:.0f} s: missing {len(missing)}, unexpected {len(unexpected)}", flush=True)

    clip = comfy.sd.CLIP(no_init=True)
    clip.cond_stage_model = te
    clip.tokenizer = mmte.MiniMaxH3Tokenizer(embedding_directory=a.embeddings)
    dev = mm.get_torch_device()
    clip.patcher = comfy.model_patcher.ModelPatcher(te, load_device=dev, offload_device=torch.device("cpu"))
    clip.layer_idx = None
    clip.use_clip_schedule = False
    clip.apply_hooks_to_conds = None
    clip.tokenizer_options = {}

    images = []
    for p in a.picture:
        im = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
        if im.shape[0] % 32 or im.shape[1] % 32:
            sys.exit(f"{p}: {im.shape[1]}x{im.shape[0]} - sides must be multiples of 32")
        images.append(torch.from_numpy(im)[None])
    t0 = time.time()
    tokens = clip.tokenize(a.prompt, images=images)
    c = clip.encode_from_tokens_scheduled(tokens)
    ctx = c[0][0].float().cpu().contiguous()
    tags = c[0][1].get("minimax_token_tags")
    tags = torch.ones(ctx.shape[1], dtype=torch.int32) if tags is None else tags.to(torch.int32).cpu().contiguous()
    print(f"encoded in {time.time() - t0:.0f} s: context {tuple(ctx.shape)}, {int((tags == 0).sum())} vision-tagged tokens", flush=True)
    save_file({"context": ctx, "token_tags": tags}, a.out, metadata={"prompt": a.prompt, "pictures": ",".join(a.picture)})
    print("written", a.out)


if __name__ == "__main__":
    main()
