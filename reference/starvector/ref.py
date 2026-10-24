"""StarVector-1B im2svg in its own code (the starvector package, transformers 4.49), on the CPU in float32 - the
reference for nextcuda's StarVector (image/starvector): each stage's tensors, dumped for `nextcuda image check`.

    CUDA_VISIBLE_DEVICES= python ref.py MODEL_DIR PICTURE --out DIR [--greedy 400] [--sample-max 0]

Run where the starvector package is installed (the platform's cp-vectorize image: /opt/vec/bin/python). MODEL_DIR is
the checkpoint nextcuda loads (ServiceNow/starvector-1b-im2svg). The base StarCoder the package builds first comes
from the checkpoint's own config and tokenizer, as the platform's container does (its weights are overwritten by the
checkpoint's right after, so none are read).

Dumps (DIR): pixels.npy [3, 224, 224] (the processor on the picture flattened on white, as the platform sends it);
enc.npy [257, 1024] (the CLIP encoder and ln_vision); adapter.npy [257, 2048] (the image projection: the embeddings
the language model starts from); prompt.npy (the prompt "<svg"'s ids) and stop.npy ("</svg>"'s, the stop sequence);
logits0.npy [vocab] (the prefill's last row); tokens.npy (greedy decoding: num_beams 1, no sampling, until "</svg>" or
--greedy new tokens) and logits.npy [min(T, 48), vocab] (the decoder teacher-forced on them: rows after each token);
greedy.txt (generate_im2svg's text for those settings); with --sample-max N, sample.txt (generate_im2svg's defaults:
2 beams sampled, top_p 0.9, max_length N; torch seed 0).
"""
import argparse
import json
import os
import shutil
import tempfile

import numpy as np
import torch
from PIL import Image

STARCODERBASE_1B = {
    "architectures": ["GPTBigCodeForCausalLM"], "model_type": "gpt_bigcode",
    "activation_function": "gelu_pytorch_tanh", "attention_softmax_in_fp32": True,
    "scale_attention_softmax_in_fp32": True, "scale_attn_weights": True,
    "attn_pdrop": 0.1, "embd_pdrop": 0.1, "resid_pdrop": 0.1, "initializer_range": 0.02,
    "layer_norm_epsilon": 1e-5, "multi_query": True, "n_embd": 2048, "n_head": 16,
    "n_inner": 8192, "n_layer": 24, "n_positions": 8192, "vocab_size": 49152,
    "bos_token_id": 0, "eos_token_id": 0, "use_cache": True,
}
BASE_VOCAB = 49152


def starcoder_base(model_dir, dest):
    """A local stand-in for bigcode/starcoderbase-1b (the platform's container does the same)."""
    os.makedirs(dest, exist_ok=True)
    with open(os.path.join(dest, "config.json"), "w") as f:
        json.dump(STARCODERBASE_1B, f)
    for name in ("vocab.json", "merges.txt"):
        shutil.copyfile(os.path.join(model_dir, name), os.path.join(dest, name))
    with open(os.path.join(model_dir, "tokenizer_config.json")) as f:
        tc = json.load(f)
    tc["added_tokens_decoder"] = {k: v for k, v in (tc.get("added_tokens_decoder") or {}).items() if int(k) < BASE_VOCAB}
    tc.pop("pad_token", None)
    tc.pop("processor_class", None)
    with open(os.path.join(dest, "tokenizer_config.json"), "w") as f:
        json.dump(tc, f)
    with open(os.path.join(model_dir, "special_tokens_map.json")) as f:
        sm = json.load(f)
    sm.pop("pad_token", None)
    with open(os.path.join(dest, "special_tokens_map.json"), "w") as f:
        json.dump(sm, f)
    return dest


def from_config_for(auto_cls, base):
    orig = auto_cls.from_pretrained.__func__

    def from_pretrained(cls, name, *args, **kw):
        if str(name) != base:
            return orig(cls, name, *args, **kw)
        return cls.from_config(kw["config"], torch_dtype=torch.float32)

    auto_cls.from_pretrained = classmethod(from_pretrained)


def flatten_on_white(img):
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    bg.alpha_composite(img)
    return bg.convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("picture")
    ap.add_argument("--out", required=True)
    ap.add_argument("--greedy", type=int, default=400)
    ap.add_argument("--sample-max", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_grad_enabled(False)
    from transformers import AutoModelForCausalLM

    work = tempfile.mkdtemp(prefix="sv-ref-")
    base = starcoder_base(a.model_dir, os.path.join(work, "starcoderbase-1b"))
    from_config_for(AutoModelForCausalLM, base)
    model = AutoModelForCausalLM.from_pretrained(a.model_dir, torch_dtype=torch.float32, trust_remote_code=True, starcoder_model_name=base)
    model.float().eval()
    sv = model.model
    sv.model_precision = torch.float32
    tok = sv.svg_transformer.tokenizer
    lm = sv.svg_transformer.transformer
    # GPTBigCode ties its output to the token embedding (the checkpoint has no lm_head): under transformers 4.49 the
    # tie is lost while the checkpoint loads (the head is materialised empty - zeros in float32, uninitialised memory in
    # half, as the platform's container loads it) - tied again here, as the architecture is
    lm.lm_head.weight = lm.transformer.wte.weight

    img = flatten_on_white(Image.open(a.picture).convert("RGBA"))
    pixels = sv.processor(img, return_tensors="pt")["pixel_values"]
    while pixels.dim() > 4:
        pixels = pixels.squeeze(0)
    if pixels.dim() == 3:
        pixels = pixels.unsqueeze(0)
    pixels = pixels.float()
    np.save(os.path.join(a.out, "pixels.npy"), pixels[0].numpy().astype(np.float32))
    enc = sv.image_encoder(pixels)
    np.save(os.path.join(a.out, "enc.npy"), enc[0].numpy().astype(np.float32))
    emb = sv.image_projection(enc)
    np.save(os.path.join(a.out, "adapter.npy"), emb[0].numpy().astype(np.float32))
    prompt = tok("<svg", add_special_tokens=False)["input_ids"]
    stop = tok("</svg>", add_special_tokens=False)["input_ids"]
    np.save(os.path.join(a.out, "prompt.npy"), np.array(prompt, dtype=np.int64))
    np.save(os.path.join(a.out, "stop.npy"), np.array(stop, dtype=np.int64))
    wte = lm.transformer.wte
    x = torch.cat([emb, wte(torch.tensor([prompt]))], dim=1)
    out = lm(inputs_embeds=x, use_cache=True)
    np.save(os.path.join(a.out, "logits0.npy"), out.logits[0, -1].numpy().astype(np.float32))
    # greedy, by hand: the same decisions generate makes with num_beams 1 and no sampling (argmax, stop on "</svg>")
    past = out.past_key_values
    nxt = int(out.logits[0, -1].argmax())
    toks = []
    for _ in range(a.greedy):
        toks.append(nxt)
        if toks[-len(stop):] == stop or nxt == tok.eos_token_id:
            break
        o = lm(input_ids=torch.tensor([[nxt]]), past_key_values=past, use_cache=True)
        past = o.past_key_values
        nxt = int(o.logits[0, -1].argmax())
    np.save(os.path.join(a.out, "tokens.npy"), np.array(toks, dtype=np.int64))
    rows = min(len(toks), 48)
    tf = lm(inputs_embeds=torch.cat([x, wte(torch.tensor([toks[:rows]]))], dim=1)).logits[0, x.shape[1] - 1: x.shape[1] - 1 + rows]
    np.save(os.path.join(a.out, "logits.npy"), tf.numpy().astype(np.float32))
    with open(os.path.join(a.out, "greedy.txt"), "w") as f:
        f.write(tok.decode(prompt + toks, skip_special_tokens=True))
    if a.sample_max:
        torch.manual_seed(0)
        txt = sv.generate_im2svg({"image": pixels}, max_length=a.sample_max)[0]
        with open(os.path.join(a.out, "sample.txt"), "w") as f:
            f.write(txt)
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump({"picture": os.path.basename(a.picture), "prompt": prompt, "stop": stop, "greedy_tokens": len(toks),
                   "eos": tok.eos_token_id, "pad": tok.pad_token_id, "vocab": len(tok)}, f)
    print(f"dumped to {a.out}: {len(toks)} greedy tokens")


if __name__ == "__main__":
    main()
