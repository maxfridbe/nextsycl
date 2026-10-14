# The reference stages of PaddleOCR-VL (transformers, the original weights, float32, CPU) as a dump our check reads:
# <dir>/<name>.f32 + index.tsv (name, type, ne0..ne3 innermost first), tokens (input ids), positions (3 x L), gen.
#   python dump.py PICTURE PROMPT DIR [N]
import os, sys, torch, numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
d = "/var/mnt/2TBSSD/nextsycl-ref/paddleocr-vl"
pic, prompt, out = sys.argv[1], sys.argv[2], sys.argv[3]
n_gen = int(sys.argv[4]) if len(sys.argv) > 4 else 32
os.makedirs(out, exist_ok=True)
proc = AutoProcessor.from_pretrained(d)
model = AutoModelForImageTextToText.from_pretrained(d, torch_dtype=torch.float32).eval()
img = Image.open(pic).convert("RGB")
msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
idx = open(os.path.join(out, "index.tsv"), "w")
def put(name, t):
    a = t.detach().float().contiguous().numpy()
    a.tofile(os.path.join(out, name + ".f32"))
    ne = list(a.shape[::-1]) + [1] * (4 - a.ndim)
    idx.write("%s\tf32\t%d\t%d\t%d\t%d\n" % (name, *ne[:4])); idx.flush()
put("pixel_values", inputs["pixel_values"])
open(os.path.join(out, "grid.txt"), "w").write(" ".join(str(int(v)) for v in inputs["image_grid_thw"][0]))
open(os.path.join(out, "tokens.txt"), "w").write(" ".join(str(int(v)) for v in inputs["input_ids"][0]))
m = model.model
hooks = []
vis = m.visual.vision_model
hooks.append(vis.embeddings.register_forward_hook(lambda mod, i, o: put("v_embd", o)))
for li, layer in enumerate(vis.encoder.layers):
    hooks.append(layer.register_forward_hook(lambda mod, i, o, li=li: put(f"v_layer-{li}", o[0] if isinstance(o, tuple) else o)))
hooks.append(vis.post_layernorm.register_forward_hook(lambda mod, i, o: put("v_out", o)))
proj = m.projector if hasattr(m, "projector") else m.mlp_AR
hooks.append(proj.register_forward_hook(lambda mod, i, o: put("projected", o)))
lm = m.language_model
for li, layer in enumerate(lm.layers):
    hooks.append(layer.register_forward_hook(lambda mod, i, o, li=li: put(f"l_out-{li}", o[0] if isinstance(o, tuple) else o)))
pos, _ = m.get_rope_index(inputs["input_ids"], inputs["mm_token_type_ids"], inputs["image_grid_thw"], attention_mask=inputs["attention_mask"])
np.savetxt(os.path.join(out, "positions.txt"), pos[:, 0, :].numpy(), fmt="%d")
with torch.no_grad():
    o = model(**inputs)
put("logits_last", o.logits[0, -1])
for h in hooks: h.remove()
with torch.no_grad():
    g = model.generate(**inputs, use_cache=True, max_new_tokens=n_gen, do_sample=False)
gen = g[0, inputs["input_ids"].shape[1]:]
open(os.path.join(out, "gen.txt"), "w").write("\n".join(str(int(v)) for v in gen))
print(proc.decode(gen, skip_special_tokens=False))
print({k: tuple(v.shape) for k, v in inputs.items()}, "grid", inputs["image_grid_thw"].tolist(), file=sys.stderr)
