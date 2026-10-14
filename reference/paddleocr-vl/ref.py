# The reference: the transformers implementation of PaddleOCR-VL (the original BF16 weights, float32 on the CPU), greedy.
#   python ref.py PICTURE PROMPT [MAX_NEW]
import sys, time, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
d = "/var/mnt/2TBSSD/nextsycl-ref/paddleocr-vl"
proc = AutoProcessor.from_pretrained(d)
model = AutoModelForImageTextToText.from_pretrained(d, torch_dtype=torch.float32).eval()
img = Image.open(sys.argv[1]).convert("RGB")
msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": sys.argv[2]}]}]
inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
print({k: tuple(v.shape) for k, v in inputs.items()}, file=sys.stderr)
t0 = time.time()
with torch.no_grad():
    out = model.generate(**inputs, use_cache=True, max_new_tokens=int(sys.argv[3]) if len(sys.argv) > 3 else 1024, do_sample=False)
gen = out[0, inputs["input_ids"].shape[1]:]
print(proc.decode(gen, skip_special_tokens=False))
print(f"[{len(gen)} tokens in {time.time()-t0:.1f} s]", file=sys.stderr)
